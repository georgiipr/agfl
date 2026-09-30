"""Aggregation and strictly matched seed/split comparisons from saved runs."""
from collections import defaultdict
import csv
import itertools
import json
import math
from pathlib import Path
import warnings
import numpy as np
from scipy import stats
from .storage import write_json
from .result_paths import result_paths
from .config import comparison_identity, experiment_identity, experiment_selection, digest

METRICS = ('accuracy', 'roc_auc', 'f1')
PRIMARY = {'mha', 'performer', 'linformer', 'nystromformer'}


def run_graph_axis(run):
    """Read recorded semantics; never relabel an old EEG time graph as spatial."""
    return run.get('token_axis') or run['config'].get('model_options', {}).get('attention_axis', 'unspecified')


def attention_display_name(key, options=None):
    """Readable variant labels; raw method keys and saved identities stay intact."""
    options = options or {}
    if key in {'agfl', 'mha'}:
        additions = [label for flag, label in (('temporal_bias', 'temporal bias'), ('output_gate', 'output gate'))
                     if options.get(flag, False)]
        if additions:
            original = {k: v for k, v in options.items() if k not in {'temporal_bias', 'output_gate'}}
            return attention_display_name(key, original) + ' + ' + ' + '.join(additions)
    if key == 'agfl':
        if options.get('coefficient_conditioning', 'static') == 'feature_contrast':
            scale = options.get('coefficient_conditioning_scale', .5)
            return f'AGFL (feature routing, scale {scale:g})'
        if options.get('coefficient_conditioning', 'static') == 'token_contrast':
            scale = options.get('coefficient_conditioning_scale', .5)
            return f'AGFL (token routing, scale {scale:g})'
        if options.get('coefficient_conditioning', 'static') == 'trial_power':
            scale = options.get('coefficient_conditioning_scale', .5)
            return f'AGFL (trial-conditioned, scale {scale:g})'
        return 'AGFL (static)'
    if key == 'linformer' and 'projection_rank' in options:
        return f"Linformer (rank {options['projection_rank']})"
    if key == 'nystromformer' and 'landmarks' in options:
        return f"Nyströmformer ({options['landmarks']} landmarks)"
    return {'mha': 'MHA', 'performer': 'Performer',
            'linformer': 'Linformer', 'nystromformer': 'Nyströmformer'}.get(key, key)


def conditioning_control(candidate, control):
    """Pair conditioning ablations with matched static/token controls."""
    left, right = candidate.get('attention_options', {}), control.get('attention_options', {})
    left_mode = left.get('coefficient_conditioning', 'static')
    right_mode = right.get('coefficient_conditioning', 'static')
    static_control = (left_mode in {'trial_power', 'token_contrast', 'feature_contrast'}
                      and right_mode == 'static')
    token_control = (left_mode == 'feature_contrast' and right_mode == 'token_contrast'
                     and left.get('coefficient_conditioning_scale', .5)
                     == right.get('coefficient_conditioning_scale', .5))
    if not (static_control or token_control):
        return False
    ignored = {'coefficient_conditioning', 'coefficient_conditioning_scale'}
    return ({k: v for k, v in left.items() if k not in ignored}
            == {k: v for k, v in right.items() if k not in ignored})


def paired_statistics(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Paired samples must have equal lengths and finite values')
    d = a - b
    result = {'n_pairs': len(d), 'mean_delta': float(d.mean()) if len(d) else None,
              't_statistic': None, 't_pvalue': None, 'wilcoxon_statistic': None,
              'wilcoxon_pvalue': None, 'cohen_dz': None, 'rank_biserial': None, 'reason': None}
    if len(d) < 2:
        result['reason'] = 'At least two matched pairs are required'
        return result
    sd = d.std(ddof=1)
    if np.all(d == 0):
        result.update(t_statistic=0., t_pvalue=1., wilcoxon_statistic=0., wilcoxon_pvalue=1., rank_biserial=0.,
                      reason='All paired differences are zero; dz is undefined')
        return result
    if sd > 1e-15:
        t = stats.ttest_rel(a, b)
        result.update(t_statistic=float(t.statistic), t_pvalue=float(t.pvalue), cohen_dz=float(d.mean() / sd))
    else:
        result['reason'] = 'Constant nonzero paired difference: t statistic and dz are undefined'
    # Quantize numerical subtraction noise before rank ties; precision is disclosed.
    d_rank = np.round(d, decimals=12)
    if np.all(d_rank == 0):
        result.update(wilcoxon_statistic=0., wilcoxon_pvalue=1., rank_biserial=0.)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            w = stats.wilcoxon(d_rank, zero_method='wilcox', alternative='two-sided', method='auto')
        nonzero = d_rank[d_rank != 0]
        ranks = stats.rankdata(abs(nonzero))
        result.update(wilcoxon_statistic=float(w.statistic), wilcoxon_pvalue=float(w.pvalue),
                      rank_biserial=float(np.sum(ranks * np.sign(nonzero)) / np.sum(ranks)))
    return result


def holm(values):
    adjusted = [None] * len(values)
    valid = sorted((p, i) for i, p in enumerate(values) if p is not None)
    running = 0.
    for rank, (p, index) in enumerate(valid):
        running = max(running, min(1., (len(valid) - rank) * p))
        adjusted[index] = running
    return adjusted


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys or ['no_results'])
        writer.writeheader()
        writer.writerows(rows)


def class_statistics(runs):
    """Keep individual class errors visible; average rates over seeds, not trials."""
    rows, groups = [], defaultdict(list)
    for run in runs:
        names = run['config'].get('resolved_metadata', {}).get('label_names', [])
        for partition in ('validation', 'test'):
            entries = run[partition].get('per_class', [])
            # Older reports have no per-class records. Do not invent them or
            # silently count missing classes as perfect/zero performance.
            for index, entry in enumerate(entries):
                if entry['class_id'] != index:
                    raise ValueError('Per-class metric IDs must be ordered and contiguous')
                row = {key: run[key] for key in ('experiment_id', 'comparison_id', 'dataset', 'seed', 'split_id')}
                row.update(model=run['backbone_key'], attention=run['attention_key'],
                           attention_label=attention_display_name(run['attention_key'], run['config'].get('attention_options', {})),
                           subject_id=run['config'].get('subject_id'), partition=partition,
                           class_name=names[index] if len(names) == len(entries) else f'Class {index}', **entry)
                for metric in ('recall', 'precision', 'f1'):
                    value = row[metric]
                    if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                        raise ValueError('Per-class rates must be finite in [0, 1] or null')
                rows.append(row)
                groups[(run['experiment_id'], partition, index)].append(row)
    summaries = []
    for _, records in sorted(groups.items()):
        if len({row['class_name'] for row in records}) != 1:
            raise ValueError('Class names differ across seeds')
        row = {key: records[0][key] for key in (
            'experiment_id', 'comparison_id', 'dataset', 'model', 'attention',
            'attention_label', 'subject_id', 'partition', 'class_id', 'class_name')}
        row.update(n_seeds=len(records), support_occurrences=sum(r['support'] for r in records))
        for metric in ('recall', 'precision', 'f1'):
            values = [r[metric] for r in records if r[metric] is not None]
            row.update({metric + '_mean': float(np.mean(values)) if values else None,
                        metric + '_std': float(np.std(values, ddof=1)) if len(values) > 1 else None,
                        metric + '_n': len(values)})
        summaries.append(row)
    return rows, summaries


def subject_aggregates(runs):
    """Equal-weight subject means; never treat subject x seed as independent people."""
    cohorts = defaultdict(list)
    for run in runs:
        config = run['config']
        if not config.get('subject_id'):
            continue
        settings = {key: config[key] for key in (
            'dataset', 'model', 'attention', 'model_variant', 'model_options',
            'attention_options', 'training', 'split', 'comparison_family',
            'deterministic', 'device', 'threads')}
        settings['data'] = {k: v for k, v in config['data'].items() if k != 'subjects'}
        settings['provenance'] = {k: config.get('provenance', {}).get(k) for k in ('source_sha256', 'packages')}
        if 'study' in config:
            settings.update(study=config['study'], study_arm=config['study_arm'])
        cohorts[digest(settings)[:20]].append(run)
    output = []
    for cohort, records in sorted(cohorts.items()):
        subjects = sorted({r['config']['subject_id'] for r in records})
        seed_sets = {s: sorted(r['seed'] for r in records if r['config']['subject_id'] == s) for s in subjects}
        if any(len(seeds) != len(set(seeds)) for seeds in seed_sets.values()):
            raise ValueError('Duplicate subject/seed in a subject summary')
        # Different seed coverage is disclosed and must not produce a deceptively
        # complete overall score; individual rows remain available.
        same_seeds = all(seeds == seed_sets[subjects[0]] for seeds in seed_sets.values())
        plan = records[0]['config'].get('study')
        declared_complete = (plan is None or (set(subjects) == set(plan['subjects'])
                             and all(set(seeds) == set(plan['seeds']) for seeds in seed_sets.values())))
        for partition, metric in itertools.product(('validation', 'test'), METRICS):
            means = []
            for subject in subjects:
                values = [r[partition].get(metric) for r in records if r['config']['subject_id'] == subject]
                if all(v is not None for v in values):
                    means.append(float(np.mean(values)))
            complete = same_seeds and declared_complete and len(means) == len(subjects)
            output.append({
                'cohort_id': cohort, 'dataset': records[0]['dataset'], 'model': records[0]['backbone_key'],
                'attention': records[0]['attention_key'],
                'token_axis': run_graph_axis(records[0]),
                'attention_label': attention_display_name(records[0]['attention_key'], records[0]['config'].get('attention_options', {})),
                'partition': partition, 'metric': metric,
                'subjects': subjects, 'n_subjects': len(subjects), 'seeds_by_subject': seed_sets,
                'complete_seed_coverage': same_seeds,
                'complete_declared_coverage': declared_complete,
                'mean_of_subject_means': float(np.mean(means)) if complete else None,
                'between_subject_sd': float(np.std(means, ddof=1)) if complete and len(means) > 1 else None,
                'reason': None if complete else 'Missing declared subjects/seeds, unequal seed coverage or undefined metrics',
            })
    return output


def analyze_results(input_root, output_dir):
    paths = result_paths(input_root)
    if not paths:
        raise ValueError(f'No saved result.json files under {input_root}')
    runs, groups = [], defaultdict(list)
    seen = set()
    for path in paths:
        if path.with_name('failure.json').exists():
            continue
        r = json.loads(path.read_text())
        if r.get('status') != 'completed':
            continue
        required = {'dataset', 'model', 'model_variant', 'seed', 'split_id', 'dataset_fingerprint',
                    'comparison_id', 'experiment_id', 'parameter_count', 'test', 'validation', 'config'}
        if not required <= r.keys():
            raise ValueError(f'Incomplete result schema: {path}')
        if r['experiment_id'] != experiment_identity(r['config']) or r['comparison_id'] != comparison_identity(r['config']):
            raise ValueError(f'Result identities disagree with the saved configuration: {path}')
        if any(r[key] != r['config'][key] for key in ('dataset', 'model', 'model_variant', 'dataset_fingerprint')):
            raise ValueError(f'Result metadata disagrees with the saved configuration: {path}')
        if r['config'].get('schema_version', 1) >= 2 and r.get('attention') != r['config']['attention']:
            raise ValueError(f'Result attention differs from saved configuration: {path}')
        if r['config']['seeds'] != [r['seed']] or r['config']['expected_split_id'] != r['split_id']:
            raise ValueError(f'Result seed/split differs from the saved configuration: {path}')
        if r.get('subject_id') != r['config'].get('subject_id'):
            raise ValueError(f'Result subject differs from the saved configuration: {path}')
        identity = r['experiment_id'], r['seed']
        if identity in seen:
            raise ValueError(f'Duplicate experiment/seed would create pseudoreplication: {path}')
        seen.add(identity)
        r['source'] = str(path.resolve())
        r['backbone_key'], r['attention_key'] = experiment_selection(r['config'])
        runs.append(r)
        groups[r['experiment_id']].append(r)
    if not runs:
        raise ValueError('No completed experiments')
    summaries, flat = [], []
    for experiment, records in sorted(groups.items()):
        first = records[0]
        if len({r['parameter_count'] for r in records}) != 1:
            raise ValueError(f'Parameter counts differ within {experiment}')
        if len({r['comparison_id'] for r in records}) != 1:
            raise ValueError(f'Comparison protocol differs within {experiment}')
        provenance = {r['config'].get('provenance', {}).get('source_sha256') for r in records}
        if len(provenance) > 1:
            raise ValueError(f'Source code differs across seeds within {experiment}')
        summary = {k: first[k] for k in ('dataset', 'model', 'model_variant', 'experiment_id', 'comparison_id', 'parameter_count')}
        summary.update(model=first['backbone_key'], attention=first['attention_key'], subject_id=first['config'].get('subject_id'))
        summary.update(token_axis=run_graph_axis(first), num_tokens=first.get('num_tokens'))
        summary.update(n_seeds=len(records), seeds=sorted(r['seed'] for r in records),
                       model_options=first['config']['model_options'],
                       attention_options=first['config'].get('attention_options', {}), metrics={})
        summary['attention_label'] = attention_display_name(
            first['attention_key'], first['config'].get('attention_options', first['config']['model_options']))
        row = {k: v for k, v in summary.items() if k not in {'metrics', 'model_options', 'attention_options', 'seeds'}}
        row['seeds'] = json.dumps(summary['seeds'])
        row['model_options'] = json.dumps(summary['model_options'], sort_keys=True)
        row['attention_options'] = json.dumps(summary['attention_options'], sort_keys=True)
        for partition, metric in itertools.product(('validation', 'test'), METRICS):
            values = [r[partition][metric] for r in records if r[partition].get(metric) is not None]
            if any(not math.isfinite(v) for v in values):
                raise ValueError('Saved metric is nonfinite; use null for undefined metrics')
            stats_ = {'mean': float(np.mean(values)) if values else None,
                      'std': float(np.std(values, ddof=1)) if len(values) > 1 else None, 'n': len(values)}
            summary['metrics'][f'{partition}_{metric}'] = stats_
            for name, value in stats_.items():
                row[f'{partition}_{metric}_{name}'] = value
        summaries.append(summary)
        flat.append(row)
    comparisons = []
    from .temporal_gate_study import temporal_gate_control, summarize_temporal_gate, write_study_report
    for agfl_id, agfl_runs in groups.items():
        if agfl_runs[0]['attention_key'] != 'agfl':
            continue
        for baseline_id, baseline_runs in groups.items():
            first = baseline_runs[0]
            is_control = (first['attention_key'] == 'agfl'
                          and conditioning_control(agfl_runs[0]['config'], first['config']))
            is_temporal_gate = (first['attention_key'] == 'agfl'
                                and temporal_gate_control(agfl_runs[0]['config'], first['config']))
            if (not (first['attention_key'] in PRIMARY or is_control or is_temporal_gate)
                    or first['backbone_key'] != agfl_runs[0]['backbone_key']
                    or first['comparison_id'] != agfl_runs[0]['comparison_id']):
                continue
            def pair_key(r):
                return r['seed'], r['split_id'], r['dataset_fingerprint']
            a_map = {pair_key(r): r for r in agfl_runs}
            b_map = {pair_key(r): r for r in baseline_runs}
            common = sorted(a_map.keys() & b_map.keys())
            for metric in METRICS:
                pairs = [(a_map[k]['test'].get(metric), b_map[k]['test'].get(metric)) for k in common]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                stat = paired_statistics([a for a, b in pairs], [b for a, b in pairs])
                stat.update(dataset=first['dataset'], model=first['backbone_key'], model_variant=first['model_variant'],
                            subject_id=first['config'].get('subject_id'),
                            agfl_experiment_id=agfl_id, baseline_experiment_id=baseline_id,
                            baseline=first['attention_key'], metric=metric, comparison_id=first['comparison_id'],
                            comparison_kind=(('output_gate' if run_graph_axis(first) == 'electrode' else 'temporal_gate') if is_temporal_gate else
                                             'coefficient_conditioning' if is_control else 'attention'),
                            agfl_label=attention_display_name('agfl', agfl_runs[0]['config'].get('attention_options', {})),
                            baseline_label=attention_display_name(first['attention_key'], first['config'].get('attention_options', {})),
                            n_unmatched_agfl=len(a_map.keys() - b_map.keys()),
                            n_unmatched_baseline=len(b_map.keys() - a_map.keys()),
                            n_undefined=len(common)-len(pairs))
                comparisons.append(stat)
    # One declared family per test across all supplied models/ablations/metrics.
    for test in ('t', 'wilcoxon'):
        for row, adjusted in zip(comparisons, holm([r[f'{test}_pvalue'] for r in comparisons])):
            row[f'{test}_pvalue_holm'] = adjusted
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paired_ids = {row[key] for row in comparisons for key in ('agfl_experiment_id', 'baseline_experiment_id')}
    across_subjects = subject_aggregates(runs)
    class_rows, class_summaries = class_statistics(runs)
    temporal_gate = summarize_temporal_gate(runs)
    from .multi_subject_study import summarize_subject_studies, write_subject_study_reports
    subject_studies = summarize_subject_studies(runs)
    result = {'schema_version': 2, 'n_runs': len(runs), 'experiments': summaries, 'comparisons': comparisons,
              'across_subjects': across_subjects,
              'per_class': class_summaries,
              'temporal_gate_studies': temporal_gate,
              'subject_studies': subject_studies,
              'experiments_without_comparable_counterpart': sorted(set(groups) - paired_ids),
              'sources': [r['source'] for r in runs],
              'methodology': 'Sample SD (ddof=1); paired AGFL minus attention baseline, plus matched conditioning and temporal/gate ablations when present; two-sided tests; Wilcoxon differences rounded to 12 decimals; Holm family includes all supplied comparisons and metrics separately per test. Repeated overlapping splits are dependent, so seed-level inference is exploratory. Temporal/gate architecture ranking uses validation only; interaction is descriptive.'}
    write_json(output / 'aggregation.json', result)
    write_csv(output / 'per_model.csv', flat)
    write_csv(output / 'per_subject.csv', [r for r in flat if r.get('subject_id')])
    write_csv(output / 'across_subjects.csv', [{**row, 'subjects': json.dumps(row['subjects']),
               'seeds_by_subject': json.dumps(row['seeds_by_subject'], sort_keys=True)} for row in across_subjects])
    write_csv(output / 'attention_comparison.csv', [r for r in flat if r['attention'] in PRIMARY | {'agfl'}])
    write_csv(output / 'agfl_ablations.csv', [r for r in flat if r['attention'] == 'agfl'])
    write_csv(output / 'statistical_comparisons.csv', comparisons)
    write_csv(output / 'per_class.csv', class_rows)
    write_csv(output / 'per_class_summary.csv', class_summaries)
    if temporal_gate:
        write_study_report(output, temporal_gate)
    if subject_studies:
        write_subject_study_reports(output, subject_studies)
    lines = ['# Saved experiment analysis', '', result['methodology'], '',
             '| Dataset | Subject | Model | Graph nodes | Attention | Seeds | Parameters | Accuracy | ROC-AUC | Macro F1 |',
             '|---|---|---|---|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        formatted = []
        for metric in METRICS:
            value = row['metrics'][f'test_{metric}']
            formatted.append('undefined' if value['mean'] is None else f"{value['mean']:.4f} ± " + (f"{value['std']:.4f}" if value['std'] is not None else 'undefined'))
        lines.append(f"| {row['dataset']} | {row.get('subject_id') or 'cohort'} | {row['model']} ({row['experiment_id'][:8]}) | {row['token_axis']} | {row['attention_label']} | {row['n_seeds']} | {row['parameter_count']} | " + ' | '.join(formatted) + ' |')
    if subject_studies:
        notices = ['## Declared subject study', '',
                   '[Overall averages, subject-paired comparisons and completeness](subject_study.md).', '']
        for study in subject_studies:
            notices += [f"**{study['name']}: {'COMPLETE' if study['complete'] else 'INCOMPLETE'} — {study['completed_runs']}/{study['expected_runs']} fits.**", '']
        # Put cohort coverage before the long per-subject/seed tables.
        lines[4:4] = notices
    if across_subjects:
        lines += ['', '## Equal-weight subject summaries', '',
                  'Average seeds within each subject first. SD below is between subject means, not across all subject/seed runs.', '',
                  '| Model | Attention | Subjects | Metric | Test mean | Between-subject SD |',
                  '|---|---|---|---|---:|---:|']
        for row in across_subjects:
            if row['partition'] != 'test':
                continue
            mean, sd = row['mean_of_subject_means'], row['between_subject_sd']
            lines.append(f"| {row['model']} | {row['attention_label']} ({row['cohort_id'][:8]}) | {', '.join(row['subjects'])} | {row['metric']} | "
                         + ('unavailable' if mean is None else f'{mean:.4f}') + ' | '
                         + ('unavailable' if sd is None else f'{sd:.4f}') + ' |')
    if class_summaries:
        lines += ['', '## Per-class test recall', '',
                  'Equal-weight mean and sample SD over seeds. Supports count repeated trial occurrences, not independent subjects. Validation and individual-seed rates are in per_class.csv and per_class_summary.csv.', '',
                  '| Model | Subject | Attention | Class | Seeds with recall | Recall mean ± SD |',
                  '|---|---|---|---|---:|---:|']
        for row in class_summaries:
            if row['partition'] != 'test':
                continue
            mean, sd = row['recall_mean'], row['recall_std']
            value = ('undefined' if mean is None else f'{mean:.4f}') + ' ± ' + ('undefined' if sd is None else f'{sd:.4f}')
            lines.append(f"| {row['model']} ({row['experiment_id'][:8]}) | {row.get('subject_id') or 'cohort'} | {row['attention_label']} | {row['class_name']} | {row['recall_n']} | {value} |")
    if temporal_gate:
        lines += ['', '[Temporal bias/output gate study: validation ranking, paired changes and interaction](temporal_gate_study.md).']
    lines += ['', 'Full configurations and pairing diagnostics are in aggregation.json; raw and Holm-adjusted p-values and effect sizes are in statistical_comparisons.csv.',
              'Synthetic fixtures and short smoke runs are validation artifacts, not estimates of scientific performance.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    return result
