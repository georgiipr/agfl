"""Declared subject/seed/attention matrix; complete, equal-weight reporting.

Study metadata accompanies each saved config, so a downloaded partial report
can identify even an entirely absent subject or attention arm. No training or
selection decisions use this module's evaluation summaries.
"""
from collections import defaultdict
import math
import statistics

from .config import digest

METRICS = ('accuracy', 'roc_auc', 'f1')


def validate_study_config(config):
    plan = config.get('study')
    required = {'name', 'subjects', 'seeds', 'arms'}
    if (not isinstance(plan, dict) or not required <= set(plan)
            or set(plan) - required - {'comparisons'}):
        raise ValueError('study requires name, subjects, seeds and arms; comparisons is optional')
    if not isinstance(plan['name'], str) or not plan['name'].strip():
        raise ValueError('study.name must be a nonempty string')
    for key in ('subjects', 'arms'):
        values = plan[key]
        if (not isinstance(values, list) or not values
                or any(not isinstance(v, str) or not v.strip() for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f'study.{key} must be unique nonempty strings')
    if 'comparisons' in plan:
        pairs = plan['comparisons']
        if (not isinstance(pairs, list) or not pairs
                or any(not isinstance(p, list) or len(p) != 2
                       or any(not isinstance(a, str) or a not in plan['arms'] for a in p)
                       or p[0] == p[1] for p in pairs)):
            raise ValueError('study.comparisons must contain [candidate, baseline] pairs of distinct declared arms')
        if len({frozenset(p) for p in pairs}) != len(pairs):
            raise ValueError('study.comparisons cannot duplicate a pair, including its reverse')
    seeds = plan['seeds']
    if (not isinstance(seeds, list) or not seeds
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('study.seeds must be unique nonnegative integer seeds')
    if config.get('study_arm') not in plan['arms']:
        raise ValueError('study_arm must identify a declared study arm')
    if not set(config['seeds']) <= set(seeds):
        raise ValueError('Run seeds must be a subset of study.seeds; update the study declaration explicitly')
    subject = config.get('subject_id')
    if subject is not None and subject not in plan['subjects']:
        raise ValueError('Run subject_id is not declared in study.subjects')
    data = config['data']
    if data.get('cohort') != 'individual':
        # Subjects are the units of a declared study; a pooled model has none.
        raise ValueError('A declared study requires data.cohort=individual')
    subjects = data['subjects']
    if (not isinstance(subjects, list) or any(type(s) is not int for s in subjects)
            or not {f'S{s:02d}' for s in subjects} <= set(plan['subjects'])):
        raise ValueError('data.subjects must belong to study.subjects')


def _recipe(config, *, attention):
    """Normalize subject-specific fields only; retain the shared protocol."""
    keys = ('dataset', 'model', 'model_variant', 'model_options', 'training',
            'split', 'comparison_family', 'deterministic', 'device', 'threads')
    result = {key: config[key] for key in keys}
    result['data'] = {k: v for k, v in config['data'].items() if k != 'subjects'}
    result['provenance'] = {k: config.get('provenance', {}).get(k)
                            for k in ('source_sha256', 'packages')}
    result['heads'] = config['attention_options']['heads']
    if attention:
        result.update(attention=config['attention'], attention_options=config['attention_options'])
    return digest(result)


def summarize_subject_studies(runs):
    from .analysis import attention_display_name, paired_statistics, holm, run_graph_axis
    groups = defaultdict(list)
    for run in runs:
        if 'study' in run['config'] or 'study_arm' in run['config']:
            validate_study_config(run['config'])
            groups[run['config']['study']['name']].append(run)
    output = []
    for name, records in sorted(groups.items()):
        if len({digest(r['config']['study']) for r in records}) != 1:
            raise ValueError(f'Conflicting study declarations for {name}')
        plan = records[0]['config']['study']
        subjects, seeds, arms = plan['subjects'], plan['seeds'], plan['arms']
        cells, configs = {}, {}
        for run in records:
            c = run['config']
            key = (c['study_arm'], c.get('subject_id'), run['seed'])
            if key[1] not in subjects or run['seed'] not in seeds:
                raise ValueError(f'Unexpected subject/seed in study {name}: {key}')
            if key in cells:
                raise ValueError(f'Duplicate study arm/subject/seed: {key}')
            cells[key] = run
            configs.setdefault(key[0], c)
        issues = []
        graph_axes = sorted({run_graph_axis(r) for r in records})
        if len(graph_axes) != 1:
            issues.append('Graph-node meanings differ across runs')
        graph_axis = graph_axes[0] if len(graph_axes) == 1 else 'mixed'
        if len({_recipe(r['config'], attention=False) for r in records}) != 1:
            issues.append('Shared protocol, source or numerical packages differ across runs')
        for arm in arms:
            if len({_recipe(r['config'], attention=True) for r in records
                    if r['config']['study_arm'] == arm}) > 1:
                issues.append(f'{arm}: attention settings differ across subjects/seeds')
        for subject in subjects:
            for seed in seeds:
                matching = [cells[(arm, subject, seed)] for arm in arms if (arm, subject, seed) in cells]
                identities = {(r['split_id'], r['dataset_fingerprint'], r['comparison_id']) for r in matching}
                if len(identities) > 1:
                    issues.append(f'{subject}, seed {seed}: unmatched split/data/protocol')
        missing = [{'arm': arm, 'subject': subject, 'seed': seed}
                   for arm in arms for subject in subjects for seed in seeds
                   if (arm, subject, seed) not in cells]
        labels = {arm: attention_display_name(configs[arm]['attention'], configs[arm]['attention_options'])
                  if arm in configs else arm + ' (not completed)' for arm in arms}
        per_subject, averages, mean_lookup = [], [], {}
        for arm in arms:
            for partition in ('validation', 'test'):
                for metric in METRICS:
                    means = {}
                    for subject in subjects:
                        found = [cells[(arm, subject, seed)] for seed in seeds if (arm, subject, seed) in cells]
                        values = [r[partition].get(metric) for r in found]
                        valid = (len(found) == len(seeds) and not issues
                                 and all(v is not None and math.isfinite(v) for v in values))
                        mean = float(statistics.mean(values)) if valid else None
                        if mean is not None:
                            means[subject] = mean
                        per_subject.append({'study': name, 'arm': arm, 'attention_label': labels[arm],
                                            'subject': subject, 'partition': partition, 'metric': metric,
                                            'n_seeds': len(found), 'expected_seeds': len(seeds),
                                            'complete': valid, 'mean': mean})
                    mean_lookup[(arm, partition, metric)] = means
                    complete = len(means) == len(subjects)
                    averages.append({'study': name, 'arm': arm, 'attention_label': labels[arm],
                                     'partition': partition, 'metric': metric, 'complete': complete,
                                     'n_subjects': len(means), 'expected_subjects': len(subjects),
                                     'mean': float(statistics.mean(means.values())) if complete else None,
                                     'between_subject_sd': float(statistics.stdev(means.values()))
                                     if complete and len(means) > 1 else None})
        # Explicit contrasts can compare complete AGFL revisions, rather than
        # pretending that several changed mechanisms are a one-factor ablation.
        pairs = plan.get('comparisons')
        explicit_pairs = {tuple(p) for p in pairs} if pairs is not None else None
        contrasts = []
        for candidate in arms:
            if explicit_pairs is None and candidate not in configs:
                continue
            c = configs.get(candidate)
            for baseline in arms:
                if baseline == candidate or (explicit_pairs is None and baseline not in configs):
                    continue
                if explicit_pairs is not None:
                    if (candidate, baseline) not in explicit_pairs:
                        continue
                else:
                    b = configs[baseline]
                    agfl_baseline = c['attention'] == 'agfl' and b['attention'] != 'agfl'
                    flags = ('temporal_bias', 'output_gate')
                    mha_control = (c['attention'] == b['attention'] == 'mha'
                                   and any(c['attention_options'].get(f, False) for f in flags)
                                   and not any(b['attention_options'].get(f, False) for f in flags)
                                   and {k: v for k, v in c['attention_options'].items() if k not in flags}
                                   == {k: v for k, v in b['attention_options'].items() if k not in flags})
                    agfl_control = (c['attention'] == b['attention'] == 'agfl'
                                    and c['attention_options'].get('output_gate', False)
                                    and not b['attention_options'].get('output_gate', False)
                                    and {k: v for k, v in c['attention_options'].items() if k != 'output_gate'}
                                    == {k: v for k, v in b['attention_options'].items() if k != 'output_gate'})
                    if not (agfl_baseline or mha_control or agfl_control):
                        continue
                for metric in METRICS:
                    a = mean_lookup[(candidate, 'test', metric)]
                    b_values = mean_lookup[(baseline, 'test', metric)]
                    matched = [s for s in subjects if s in a and s in b_values]
                    complete = len(matched) == len(subjects) and not missing and not issues
                    stat = paired_statistics([a[s] for s in matched], [b_values[s] for s in matched]) if complete else paired_statistics([], [])
                    if not complete:
                        stat['reason'] = 'Incomplete declared subject/seed matrix or mismatched protocol; no overall contrast computed'
                    contrasts.append({**stat, 'study': name, 'candidate': candidate, 'baseline': baseline,
                                      'candidate_label': labels[candidate], 'baseline_label': labels[baseline],
                                      'contrast_selection': 'declared' if explicit_pairs is not None else 'inferred',
                                      'metric': metric, 'complete': complete, 'matched_subjects': matched,
                                      'expected_subjects': len(subjects),
                                      'subject_deltas': {s: a[s] - b_values[s] for s in matched}})
        # Inference is withheld until the entire declared matrix is present.
        # Undefined metrics still consume a family slot (p=1).
        for test in ('t', 'wilcoxon'):
            values = [row[f'{test}_pvalue'] if row[f'{test}_pvalue'] is not None else 1. for row in contrasts]
            for row, adjusted in zip(contrasts, holm(values)):
                row[f'{test}_pvalue_holm'] = adjusted if row[f'{test}_pvalue'] is not None else None
        output.append({'name': name, 'plan': plan, 'expected_runs': len(arms) * len(subjects) * len(seeds),
                       'token_axis': graph_axis,
                       'completed_runs': len(cells), 'complete': not missing and not issues,
                       'missing_runs': missing, 'issues': issues, 'per_subject': per_subject,
                       'averages': averages, 'comparisons': contrasts,
                       'methodology': 'Average matched seeds within each subject, then weight subjects equally. SD is between subject means. Paired tests use subjects, not subject x seed rows. Holm family includes all declared contrasts and metrics separately per test. Development subjects are included; this is not wholly independent confirmation.'})
    return output


def write_subject_study_reports(output, studies):
    from .analysis import write_csv
    from .storage import write_json
    write_json(output / 'subject_study.json', studies)
    write_csv(output / 'subject_study_per_subject.csv', [row for s in studies for row in s['per_subject']])
    write_csv(output / 'subject_study_averages.csv', [row for s in studies for row in s['averages']])
    write_csv(output / 'subject_study_comparisons.csv', [row for s in studies for row in s['comparisons']])
    lines = ['# Declared multi-subject attention study', '']
    for study in studies:
        state = 'COMPLETE' if study['complete'] else 'INCOMPLETE — do not describe as the full comparison'
        lines += [f"## {study['name']}", '', f"**{state}: {study['completed_runs']}/{study['expected_runs']} fits.**", '',
                  f"Graph nodes: **{study.get('token_axis', 'unspecified')}**. Electrode weights describe sensor-level model routing, not anatomical connectivity.", '', study['methodology'], '',
                  '| Attention | Subjects complete | Accuracy | ROC-AUC | Macro F1 |',
                  '|---|---:|---:|---:|---:|']
        for arm in study['plan']['arms']:
            rows = {r['metric']: r for r in study['averages'] if r['arm'] == arm and r['partition'] == 'test'}
            values = [('unavailable' if rows[m]['mean'] is None else f"{rows[m]['mean']:.4f} ± " +
                       ('undefined' if rows[m]['between_subject_sd'] is None else f"{rows[m]['between_subject_sd']:.4f}")) for m in METRICS]
            r = rows['accuracy']
            lines.append(f"| {r['attention_label']} | {r['n_subjects']}/{r['expected_subjects']} | " + ' | '.join(values) + ' |')
        lines += ['', '## Subject-paired test comparisons', '',
                  '| Candidate − baseline | Metric | Subjects | Mean difference | t p (Holm) | Wilcoxon p (Holm) |',
                  '|---|---|---:|---:|---:|---:|']
        for row in study['comparisons']:
            fmt = lambda x: 'unavailable' if x is None else f'{x:.4f}'
            lines.append(f"| {row['candidate_label']} − {row['baseline_label']} | {row['metric']} | {row['n_pairs']} | {fmt(row['mean_delta'])} | {fmt(row['t_pvalue_holm'])} | {fmt(row['wilcoxon_pvalue_holm'])} |")
        if study['issues']:
            lines += ['', 'Protocol issues:', ''] + ['- ' + issue for issue in study['issues']]
        if study['missing_runs']:
            lines += ['', f"Missing {len(study['missing_runs'])} fits; exact arm/subject/seed entries are in subject_study.json."]
        lines += ['', 'Per-subject values: subject_study_per_subject.csv. Overall means: subject_study_averages.csv. Raw/adjusted statistics: subject_study_comparisons.csv.', '']
    (output / 'subject_study.md').write_text('\n'.join(lines) + '\n')
