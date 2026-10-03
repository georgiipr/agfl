"""Saved-result analysis for the removable 2 x 2 AGFL experiment.

No model imports. Architecture ranking uses validation accuracy, then the
validation loss at the already selected checkpoints, then parameter count.
Test predictions are descriptive and never enter ranking.
"""
from collections import defaultdict
from pathlib import Path
import json
import math

import numpy as np

from .config import digest
from .storage import write_json

FLAGS = ('temporal_bias', 'output_gate')
ARMS = {(False, False): 'control', (True, False): 'temporal_bias',
        (False, True): 'output_gate', (True, True): 'both'}
EXPECTED_SEEDS = [0, 1, 2, 3, 4]
CONTRASTS = (('temporal_bias', 'control'), ('output_gate', 'control'), ('both', 'control'),
             ('both', 'temporal_bias'), ('both', 'output_gate'))


def _state(options):
    values = tuple(options.get(flag, False) for flag in FLAGS)
    if any(type(value) is not bool for value in values):
        raise ValueError('Temporal/gate settings must be booleans in saved results')
    return values


def _shared(options):
    return {k: v for k, v in options.items() if k not in FLAGS}


def temporal_gate_control(candidate, control):
    left, right = candidate.get('attention_options', {}), control.get('attention_options', {})
    if left.get('coefficient_conditioning') != 'token_contrast' or _shared(left) != _shared(right):
        return False
    # Candidate adds one or both factors; no self/reverse/bias-vs-gate pairs.
    a, b = _state(left), _state(right)
    return a != b and all(x >= y for x, y in zip(a, b))


def _mean(values):
    return float(np.mean(values)) if values and all(v is not None and math.isfinite(v) for v in values) else None


def _prediction_changes(records, partition):
    """Validate saved predictions, then retain paired fixed/lost trial IDs."""
    from .visualization.results import read_predictions
    predictions, ids = {}, None
    for arm, run in records.items():
        root = Path(run['source']).parent
        if not all((root / name).is_file() for name in ('predictions.npz', 'split.json')):
            return {'available': False, 'reason': 'Saved predictions or split.json unavailable'}
        y, p = read_predictions(run, partition)
        with np.load(root / 'predictions.npz', allow_pickle=False) as saved:
            current_ids = saved[f'{partition}_ids'].astype(str).tolist()
        if ids is not None and (ids != current_ids or not np.array_equal(y, targets)):
            raise ValueError('Temporal/gate prediction IDs or targets differ within a matched split')
        ids, targets = current_ids, y
        predictions[arm] = p.argmax(-1)
    rows = []
    for candidate, control in CONTRASTS:
        a, b = predictions[candidate], predictions[control]
        fixed = (a == targets) & (b != targets)
        lost = (a != targets) & (b == targets)
        rows.append({'candidate': candidate, 'control': control, 'n_samples': len(targets),
                     'fixed': int(fixed.sum()), 'lost': int(lost.sum()),
                     'net_correct': int(fixed.sum() - lost.sum()),
                     'changed_predictions': int((a != b).sum()),
                     'fixed_ids': [ids[i] for i in np.flatnonzero(fixed)],
                     'lost_ids': [ids[i] for i in np.flatnonzero(lost)]})
    return {'available': True, 'contrasts': rows}


def summarize_temporal_gate(runs):
    groups = defaultdict(list)
    for run in runs:
        axis = run.get('token_axis') or run['config'].get('model_options', {}).get('attention_axis')
        if run['config'].get('model_variant') == 'eeg' and axis == 'electrode':
            # Electrode gate ablations are not missing a temporal-bias factor.
            # Their contrasts are handled by the declared interchannel study.
            continue
        options = run['config'].get('attention_options', {})
        if run['attention_key'] == 'agfl' and options.get('coefficient_conditioning') == 'token_contrast':
            groups[(run['comparison_id'], digest(_shared(options)))].append(run)
    studies = []
    for (comparison, shared), records in sorted(groups.items()):
        # One fixed AGFL configuration against MHA/Linformer is a baseline
        # comparison, not an incomplete four-arm factorial experiment.
        if len({_state(r['config']['attention_options']) for r in records}) < 2:
            continue
        if not any(any(_state(r['config']['attention_options'])) for r in records):
            continue
        arms = defaultdict(dict)
        for run in records:
            arm = ARMS[_state(run['config']['attention_options'])]
            key = run['seed'], run['split_id'], run['dataset_fingerprint']
            if key in arms[arm]:
                raise ValueError('Duplicate temporal/gate arm and matched seed')
            arms[arm][key] = run
        common = set.intersection(*(set(arms[a]) for a in ARMS.values()))
        complete = (sorted(k[0] for k in common) == EXPECTED_SEEDS
                    and all(len(arms[a]) == len(EXPECTED_SEEDS) for a in ARMS.values()))
        summaries = []
        for arm in ARMS.values():
            selected = list(arms[arm].values())
            if not selected:
                continue
            summaries.append({'arm': arm, 'experiment_id': selected[0]['experiment_id'],
                              'seeds': sorted(r['seed'] for r in selected),
                              'parameter_count': selected[0]['parameter_count'],
                              'validation_accuracy': _mean([r['validation'].get('accuracy') for r in selected]),
                              'validation_loss': _mean([r['validation'].get('loss') for r in selected]),
                              'test_accuracy': _mean([r['test'].get('accuracy') for r in selected])})
        ranking = []
        if complete and all(r['validation_accuracy'] is not None and r['validation_loss'] is not None for r in summaries):
            ranking = [r['arm'] for r in sorted(summaries, key=lambda r: (
                -r['validation_accuracy'], r['validation_loss'], r['parameter_count'], r['arm']))]
        rows, prediction_changes = [], []
        for key in sorted(common):
            matched = {arm: arms[arm][key] for arm in ARMS.values()}
            for partition in ('validation', 'test'):
                row = {'seed': key[0], 'split_id': key[1], 'partition': partition}
                for arm, run in matched.items():
                    row[arm] = run[partition]['accuracy']
                for candidate, control in CONTRASTS:
                    row[f'{candidate}_minus_{control}'] = row[candidate] - row[control]
                row['interaction'] = row['both'] - row['temporal_bias'] - row['output_gate'] + row['control']
                rows.append(row)
                prediction_changes.append({'seed': key[0], 'split_id': key[1], 'partition': partition,
                                           **_prediction_changes(matched, partition)})
        studies.append({'study_id': digest([comparison, shared])[:20], 'comparison_id': comparison,
                        'subject_id': records[0]['config'].get('subject_id'),
                        'expected_seeds': EXPECTED_SEEDS, 'complete': complete,
                        'matched_seeds': sorted(k[0] for k in common), 'arms': summaries,
                        'validation_ranking': ranking, 'selected_arm': ranking[0] if ranking else None,
                        'selection_reason': ('Validation accuracy, selected-checkpoint validation loss, then parameter count.'
                                             if ranking else 'No selection: incomplete five-seed matrix or missing validation metrics.'),
                        'per_seed': rows, 'prediction_changes': prediction_changes,
                        'interaction_definition': 'both - temporal_bias - output_gate + control (accuracy units)',
                        'note': 'Overlapping splits are dependent. Interaction and test contrasts are descriptive; no claim of significance or independent confirmation.'})
    return studies


def write_study_report(output, studies):
    from .analysis import write_csv
    output = Path(output)
    write_json(output / 'temporal_gate_study.json', studies)
    write_csv(output / 'temporal_gate_per_seed.csv', [dict(study_id=s['study_id'], **row)
                                                    for s in studies for row in s['per_seed']])
    lines = ['# AGFL temporal bias and output gate', '',
             'Four fixed arms. Architecture ranking uses validation only. All test results remain visible.', '']
    for study in studies:
        lines += [f"## {study['subject_id']} / {study['study_id']}", '',
                  f"Complete five-seed matrix: {study['complete']}. Matched seeds: {study['matched_seeds']}.", '',
                  '| Arm | Parameters | Validation accuracy | Test accuracy |', '|---|---:|---:|---:|']
        for row in study['arms']:
            fmt = lambda value: 'N/A' if value is None else f'{100 * value:.2f}%'
            lines.append(f"| {row['arm']} | {row['parameter_count']} | {fmt(row['validation_accuracy'])} | {fmt(row['test_accuracy'])} |")
        lines += ['', f"Validation-selected arm: **{study['selected_arm'] or 'not selected'}**. {study['selection_reason']}", '']
        for partition in ('validation', 'test'):
            rows = [r for r in study['per_seed'] if r['partition'] == partition]
            if rows:
                delta = 100 * _mean([r['both_minus_control'] for r in rows])
                interaction = 100 * _mean([r['interaction'] for r in rows])
                lines.append(f'{partition.capitalize()}: combined minus control **{delta:+.2f} pp**; interaction **{interaction:+.2f} pp**.')
        lines += ['', 'Interaction = both − temporal bias − output gate + control. A positive value indicates '
                  'more gain than the sum of individual changes on these splits; it is not a significance test.',
                  'Paired fixed/lost trial IDs and all five contrasts are in temporal_gate_study.json.', '', study['note'], '']
    (output / 'temporal_gate_study.md').write_text('\n'.join(lines) + '\n')


def plot_studies(writer, studies):
    for study in studies:
        figure, axes = writer.plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        for axis, partition in zip(axes, ('validation', 'test')):
            rows = [r for r in study['per_seed'] if r['partition'] == partition]
            names = list(ARMS.values())
            for row in rows:
                axis.plot(range(4), [100 * row[arm] for arm in names], 'o-', alpha=.45, label=f"Seed {row['seed']}")
            if rows:
                axis.plot(range(4), [100 * _mean([r[a] for r in rows]) for a in names], 'ks-', linewidth=2, label='Mean')
                axis.legend(fontsize=8)
            axis.set(xticks=range(4), xticklabels=['Control', 'Temporal bias', 'Output gate', 'Both'],
                     ylabel='Accuracy (%)', title=f'{partition.capitalize()} / matched splits')
            axis.tick_params(axis='x', rotation=20)
        writer.save(figure, f"temporal_gate/{study['study_id']}_accuracy", 'Paired seeds across all four arms; incomplete or mismatched seeds are excluded from these lines.')
        figure, axes = writer.plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for axis, partition in zip(axes, ('validation', 'test')):
            rows = [r for r in study['per_seed'] if r['partition'] == partition]
            axis.axhline(0, color='black', linewidth=.8)
            axis.bar([str(r['seed']) for r in rows], [100 * r['interaction'] for r in rows])
            axis.set(xlabel='Matched seed', ylabel='Interaction (percentage points)', title=partition.capitalize())
        writer.save(figure, f"temporal_gate/{study['study_id']}_interaction", 'Both − temporal bias − output gate + control. Descriptive interaction, not an independent significance test.')
