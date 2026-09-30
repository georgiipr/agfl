"""Bounded EEG comparison with electrode checks and small accuracy tables.

Execute on the cluster only. Completed metrics are collected from this
invocation, never by scanning stale results from a previous attempt.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import io
import math
from pathlib import Path
import statistics
import traceback

from .config import merge, resolve_experiments
from .presets import load_preset
from .storage import run_directory, write_json


PRESET = 'eeg-three-subject-spatial-accuracy'
MODELS = ('eegnet', 'eegencoder', 'dstseegencoder', 'conformer', 'signal_transformer')
LABELS = {'eegnet': 'EEGNet', 'eegencoder': 'EEGEncoder',
          'dstseegencoder': 'DSTS EEGEncoder', 'conformer': 'Conformer',
          'signal_transformer': 'Signal Transformer'}
SUBJECTS = ('A03', 'A04', 'A09')
ATTENTIONS = ('agfl', 'mha')
SEEDS = tuple(range(5))
EXPECTED_FITS = len(MODELS) * len(SUBJECTS) * len(ATTENTIONS) * len(SEEDS)


def study_configs(data_dir, output_dir):
    from .models import available_models, get_model_spec

    available = {key for key in available_models() if 'eeg' in get_model_spec(key).variants}
    if available != set(MODELS):
        raise ValueError('The EEG model registry changed; update the declared accuracy matrix explicitly')
    document = load_preset(PRESET)
    base = merge(document['base'], {'data': {'data_dir': str(data_dir)},
                                    'output_dir': str(output_dir)})
    configs = [c for arm in document['experiments']
               for c in resolve_experiments(merge(base, arm))]
    lookup = {(c['model'], c['attention'], c['subject_id']): c for c in configs}
    expected = {(m, a, s) for m in MODELS for a in ATTENTIONS for s in SUBJECTS}
    if len(configs) != len(lookup) or set(lookup) != expected:
        raise ValueError('Accuracy preset must contain every declared model/attention/subject exactly once')
    for c in configs:
        if (c['seeds'] != list(SEEDS) or c['dataset'] != 'eeg'
                or c['data']['subjects'] != [int(c['subject_id'][1:])]
                or c['training']['save_checkpoints'] or c['device'] != 'cuda'
                or c['model_options'].get('attention_axis', 'electrode') != 'electrode'
                or c['attention_options'].get('temporal_bias', False)):
            raise ValueError('Accuracy preset violates subject, seed, device, storage or electrode requirements')
    for m in MODELS:
        for s in SUBJECTS:
            left, right = lookup[m, 'agfl', s], lookup[m, 'mha', s]
            for key in ('data', 'split', 'training', 'model_options', 'deterministic', 'threads'):
                if left[key] != right[key]:
                    raise ValueError(f'{m}/{s}: unmatched {key}')
            if left['attention_options']['heads'] != right['attention_options']['heads']:
                raise ValueError(f'{m}/{s}: unmatched attention heads')
    return lookup


def check_electrode_models(configs):
    """Cluster preflight: check actual attention inputs, not only axis labels."""
    import torch
    from .models import get_model_spec
    from .reproducibility import seed_everything

    metadata = {'modality': 'eeg', 'channels': 22, 'samples': 1000, 'num_classes': 4}
    checks = []
    for name in MODELS:
        for attention in ATTENTIONS:
            c = configs[name, attention, SUBJECTS[0]]
            seed_everything(0, c['deterministic'], c['threads'])
            model = get_model_spec(name).build(c['model_options'], metadata,
                        attention, c['attention_options']).to(c['device']).eval()
            layers = {key: layer for key, layer in model.named_modules()
                      if getattr(layer, 'is_attention', False)}
            if not layers or model.token_axis != 'electrode' or model.num_tokens != 22:
                raise ValueError(f'{name}/{attention}: missing electrode graph')
            observed, hooks = {}, []

            def capture(key):
                def hook(layer, args, output):
                    inputs = args[0]
                    if (layer.token_axis != 'electrode' or layer.num_tokens != 22
                            or inputs.ndim != 3 or tuple(inputs.shape[:2]) != (2, 22)
                            or output.shape != inputs.shape):
                        raise ValueError(f'{name}/{attention}/{key}: attention must consume [B,22,D]')
                    observed[key] = {'token_axis': layer.token_axis,
                                     'input_shape': list(inputs.shape), 'num_tokens': layer.num_tokens}
                return hook

            try:
                for key, layer in layers.items():
                    hooks.append(layer.register_forward_hook(capture(key)))
                with torch.no_grad():
                    output = model(torch.zeros(2, 22, 1000, device=c['device']))
                if output.shape != (2, 4) or not torch.isfinite(output).all() or set(observed) != set(layers):
                    raise ValueError(f'{name}/{attention}: invalid logits or unexecuted attention module')
            finally:
                for hook in hooks:
                    hook.remove()
            checks.append({'model': name, 'attention': attention, 'layers': observed})
            print(f'Electrode check passed: {name}/{attention}, {len(layers)} layers, 22 nodes each', flush=True)
            del model, layers, output
            gc.collect()
            torch.cuda.empty_cache()
    return checks


def accuracy_rows(records):
    """Equal seed means within subjects, then equal subject means; no pooling."""
    cells = {}
    for r in records:
        key = r['model'], r['attention'], r['subject_id'], r['seed']
        if (key[0] not in MODELS or key[1] not in ATTENTIONS or key[2] not in SUBJECTS
                or key[3] not in SEEDS or key in cells or r['status'] != 'completed'):
            raise ValueError(f'Unexpected, duplicate or incomplete accuracy record: {key}')
        value = r['test']['accuracy']
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Invalid accuracy: {key}')
        cells[key] = value
    rows = []
    for model in MODELS:
        row = {'model': model}
        for attention in ATTENTIONS:
            subject_means = []
            for subject in SUBJECTS:
                scores = [cells[model, attention, subject, seed] for seed in SEEDS
                          if (model, attention, subject, seed) in cells]
                mean = statistics.mean(scores) if len(scores) == len(SEEDS) else None
                row[f'{subject}_{attention}_percent'] = None if mean is None else 100 * mean
                if mean is not None:
                    subject_means.append(mean)
            row[f'mean_{attention}_percent'] = (100 * statistics.mean(subject_means)
                                               if len(subject_means) == len(SUBJECTS) else None)
        a, b = row['mean_agfl_percent'], row['mean_mha_percent']
        row['agfl_minus_mha_pp'] = None if a is None or b is None else a - b
        row['completed_fits'] = sum(key[0] == model for key in cells)
        row['expected_fits'] = len(ATTENTIONS) * len(SUBJECTS) * len(SEEDS)
        rows.append(row)
    return rows


def _write_text(path, text):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def write_accuracy_tables(report_dir, records, failures, *, status):
    rows = accuracy_rows(records)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()})
    _write_text(report_dir / 'accuracy_table.csv', stream.getvalue())
    show = lambda value: 'pending' if value is None else f'{value:.2f}'
    lines = ['# EEG electrode attention: A03, A04, A09', '',
             f'Status: **{status}**; {len(records)}/{EXPECTED_FITS} completed fits; {len(failures)} failed fits.', '',
             'Accuracy in percent. Average five seeds within each subject, then give each subject equal weight.',
             'Incomplete subject/seed groups do not produce an overall mean. Differences are percentage points.', '',
             '| Model | AGFL mean | MHA mean | AGFL − MHA | Fits |',
             '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {LABELS[r['model']]} | {show(r['mean_agfl_percent'])} | {show(r['mean_mha_percent'])} | "
                     f"{show(r['agfl_minus_mha_pp'])} | {r['completed_fits']}/{r['expected_fits']} |")
    lines += ['', '| Model | Subject | AGFL | MHA |', '|---|---|---:|---:|']
    for r in rows:
        for s in SUBJECTS:
            lines.append(f"| {LABELS[r['model']]} | {s} | {show(r[f'{s}_agfl_percent'])} | {show(r[f'{s}_mha_percent'])} |")
    lines += ['', 'EEGNet uses the spatial-fusion repair. Every attention insertion has 22 electrode nodes.',
              'Each subject is trained independently. Weights are selected using validation accuracy, then tested.',
              'No checkpoint files or plots are produced. Small run records retain configurations, splits and provenance.',
              'These are development-subject results, not an independent across-nine-subject confirmation.', '']
    _write_text(report_dir / 'accuracy_table.md', '\n'.join(lines))
    write_json(report_dir / 'accuracy_status.json', {'status': status, 'completed_fits': len(records),
              'expected_fits': EXPECTED_FITS, 'failed_fits': failures})


def run_study(data_dir, output_dir):
    import torch
    from .engine import run_experiment
    from .reproducibility import provenance
    from .session import session_activity, session_paths

    session = session_paths(output_dir, create=True)
    # Only one writer may replace the accuracy table for this session.
    with session_activity(session.root), run_directory(session.artifacts / '_accuracy_study'):
        records, failures, paired_data, paired_protocol = [], [], {}, {}
        write_accuracy_tables(session.report, records, failures, status='preflight')
        try:
            # Invalidate old summary files before preflight, too: a missing
            # recording on a rerun must not leave yesterday's table as current.
            for name in ('spatial_checks.json', 'accuracy_plan.json'):
                (session.report / name).unlink(missing_ok=True)
            configs = study_configs(data_dir, output_dir)
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable inside the allocated job')
            for subject in SUBJECTS:
                path = Path(data_dir).expanduser() / f'{subject}T.gdf'
                if not path.is_file():
                    raise FileNotFoundError(f'Missing dataset: {path}')
            current = provenance()
            write_json(session.report / 'accuracy_plan.json', {'preset': PRESET, 'provenance': current,
                       'models': list(MODELS), 'subjects': list(SUBJECTS), 'seeds': list(SEEDS),
                       'expected_fits': EXPECTED_FITS, 'configs': list(configs.values())})
            checks = check_electrode_models(configs)
            write_json(session.report / 'spatial_checks.json', checks)
            for model in MODELS:
                for subject in SUBJECTS:
                    for seed in SEEDS:
                        for attention in ATTENTIONS:
                            # Source/package pinning makes a mid-job code change
                            # an error instead of silently mixing implementations.
                            config = copy.deepcopy(configs[model, attention, subject])
                            config.update(seeds=[seed], provenance=current)
                            print(f'Fit {len(records) + len(failures) + 1}/{EXPECTED_FITS}: '
                                  f'{model}/{attention}/{subject}/seed_{seed}', flush=True)
                            try:
                                result, = run_experiment(config, skip_completed=False)
                                modules = result['attention_modules']
                                if (result['token_axis'] != 'electrode' or result['num_tokens'] != 22
                                        or not modules or result.get('checkpoint_retained', True)
                                        or result.get('synthetic', False)
                                        or any(m['token_axis'] != 'electrode' or m['num_tokens'] != 22 for m in modules)):
                                    raise ValueError('Finished run violates the spatial/storage study contract')
                                data_identity = result['dataset_fingerprint'], result['split_id']
                                if data_identity != paired_data.setdefault((subject, seed), data_identity):
                                    raise ValueError('Trials or preprocessing differ across models/attentions')
                                pair = model, subject, seed
                                if result['comparison_id'] != paired_protocol.setdefault(pair, result['comparison_id']):
                                    raise ValueError('AGFL and MHA have mismatched comparison protocols')
                                records.append(result)
                                print(f"Test accuracy: {100 * result['test']['accuracy']:.2f}%", flush=True)
                            except Exception as error:
                                failures.append({'model': model, 'attention': attention, 'subject': subject,
                                                 'seed': seed, 'type': type(error).__name__, 'message': str(error),
                                                 'traceback': traceback.format_exc()})
                                print(f'Fit failed: {error}', flush=True)
                            write_accuracy_tables(session.report, records, failures, status='running')
                            gc.collect()
                            torch.cuda.empty_cache()
        except BaseException:
            write_accuracy_tables(session.report, records, failures, status='interrupted')
            raise
        complete = len(records) == EXPECTED_FITS and not failures
        write_accuracy_tables(session.report, records, failures, status='complete' if complete else 'incomplete')
        print((session.report / 'accuracy_table.md').read_text(), flush=True)
        if not complete:
            raise RuntimeError('Some fits failed; accuracy table marks incomplete means as pending. See accuracy_status.json.')
    return session.report / 'accuracy_table.csv'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='../ml')
    parser.add_argument('--output-dir', default='results/eeg-A03-A04-A09-spatial-accuracy-v1')
    args = parser.parse_args(argv)
    result = run_study(args.data_dir, args.output_dir)
    print(f'Accuracy table: {result}', flush=True)


if __name__ == '__main__':
    main()
