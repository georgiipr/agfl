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
from .config import comparison_identity, experiment_identity, experiment_selection

METRICS = ('accuracy', 'roc_auc', 'f1')
PRIMARY = {'mha', 'performer', 'linformer', 'nystromformer'}


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


def analyze_results(input_root, output_dir):
    paths = sorted(Path(input_root).rglob('result.json'))
    if not paths:
        raise ValueError(f'No saved result.json files under {input_root}')
    runs, groups = [], defaultdict(list)
    seen = set()
    for path in paths:
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
        summary.update(model=first['backbone_key'], attention=first['attention_key'])
        summary.update(n_seeds=len(records), seeds=sorted(r['seed'] for r in records),
                       model_options=first['config']['model_options'],
                       attention_options=first['config'].get('attention_options', {}), metrics={})
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
    for agfl_id, agfl_runs in groups.items():
        if agfl_runs[0]['attention_key'] != 'agfl':
            continue
        for baseline_id, baseline_runs in groups.items():
            first = baseline_runs[0]
            if (first['attention_key'] not in PRIMARY or first['backbone_key'] != agfl_runs[0]['backbone_key']
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
                            agfl_experiment_id=agfl_id, baseline_experiment_id=baseline_id,
                            baseline=first['attention_key'], metric=metric, comparison_id=first['comparison_id'],
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
    result = {'schema_version': 2, 'n_runs': len(runs), 'experiments': summaries, 'comparisons': comparisons,
              'experiments_without_comparable_counterpart': sorted(set(groups) - paired_ids),
              'sources': [r['source'] for r in runs],
              'methodology': 'Sample SD (ddof=1); paired AGFL minus baseline; two-sided tests; Wilcoxon differences rounded to 12 decimals; Holm family includes all supplied comparisons and metrics separately per test. Repeated overlapping splits are dependent, so seed-level inference is exploratory.'}
    write_json(output / 'aggregation.json', result)
    write_csv(output / 'per_model.csv', flat)
    write_csv(output / 'attention_comparison.csv', [r for r in flat if r['attention'] in PRIMARY | {'agfl'}])
    write_csv(output / 'agfl_ablations.csv', [r for r in flat if r['attention'] == 'agfl'])
    write_csv(output / 'statistical_comparisons.csv', comparisons)
    lines = ['# Saved experiment analysis', '', result['methodology'], '',
             '| Dataset | Model | Attention | Seeds | Parameters | Accuracy | ROC-AUC | Macro F1 |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        formatted = []
        for metric in METRICS:
            value = row['metrics'][f'test_{metric}']
            formatted.append('undefined' if value['mean'] is None else f"{value['mean']:.4f} ± " + (f"{value['std']:.4f}" if value['std'] is not None else 'undefined'))
        lines.append(f"| {row['dataset']} | {row['model']} ({row['experiment_id'][:8]}) | {row['attention']} | {row['n_seeds']} | {row['parameter_count']} | " + ' | '.join(formatted) + ' |')
    lines += ['', 'Full configurations and pairing diagnostics are in aggregation.json; raw and Holm-adjusted p-values and effect sizes are in statistical_comparisons.csv.',
              'Synthetic fixtures and short smoke runs are validation artifacts, not estimates of scientific performance.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    return result
