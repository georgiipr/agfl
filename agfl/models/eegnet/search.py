"""Bounded EEGNet/AGFL BCI IV-2a search; candidates never evaluate test."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

from agfl.config import digest, merge, resolve_experiments
from agfl.presets import load_preset


CANDIDATES = {
    'spatial_control': {
        'data': {'normalization': 'train_channel'},
        'model_options': {'temp_kernel': 32, 'f1': 16, 'f2': 32, 'pk1': 8, 'pk2': 16,
                          'batch_norm_momentum': .1, 'batch_norm_eps': 1e-5,
                          'attention_dropout': 0., 'initialization': 'pytorch'},
        'training': {'batch_size': 64, 'learning_rate': .005, 'weight_decay': .001,
                     'loss': 'focal', 'class_weights': 'balanced', 'focal_gamma': 3.,
                     'checkpoint_criterion': 'accuracy', 'gradient_clip': None,
                     'min_lr_ratio': 0., 'early_stopping_patience': 0},
    },
    'compact_train_channel': {'data': {'normalization': 'train_channel'}},
    'compact_trialnorm': {},
    'compact_recombine': {'training': {'augmentation': {'recombine_segments': 8}}},
    'compact_recombine_slow': {'training': {'learning_rate': .0005,
                                          'augmentation': {'recombine_segments': 8}}},
}


def search_configs(data_dir, output_dir, subjects, seeds, candidates=None, epochs=None):
    names = list(CANDIDATES) if candidates is None else list(candidates)
    if not names or len(set(names)) != len(names) or set(names) - CANDIDATES.keys():
        raise ValueError(f'Choose unique EEGNet candidates from {list(CANDIDATES)}')
    base = load_preset('eegnet-bci2a')
    base.update(output_dir=str(output_dir), seeds=list(seeds))
    base['data'].update(data_dir=str(data_dir), subjects=list(subjects))
    if epochs is not None:
        base['training']['epochs'] = epochs
    return {name: resolve_experiments(merge(base, CANDIDATES[name])) for name in names}


def select_candidate(candidate_runs):
    """Select within ONE subject/seed split; never pool different splits."""
    summaries, expected_seeds = [], None
    for name, records in candidate_runs.items():
        seeds = sorted(r['seed'] for r in records)
        if len(seeds) != 1 or (expected_seeds is not None and seeds != expected_seeds):
            raise ValueError('Candidate selection requires exactly one matching seed per candidate')
        expected_seeds = seeds
        if any(r['status'] != 'validation_completed' or 'test' in r for r in records):
            raise ValueError('Selection accepts only validation-only candidate artifacts')
        if any(not math.isfinite(r['validation'][key]) or not 0 <= r['validation'][key] <= 1
               for r in records for key in ('accuracy', 'f1')):
            raise ValueError('Candidate validation accuracy/F1 must be finite proportions')
        summaries.append({'candidate': name, 'seeds': seeds,
                          'validation_accuracy': statistics.mean(r['validation']['accuracy'] for r in records),
                          'validation_f1': statistics.mean(r['validation']['f1'] for r in records),
                          'epochs': [r['best_checkpoint_epoch'] for r in records]})
    if not summaries:
        raise ValueError('No candidate results')
    # F1 and then declared candidate order resolve ties. Losses from focal/CE
    # objectives are not numerically comparable and must not break ties.
    winner = max(summaries, key=lambda r: (r['validation_accuracy'], r['validation_f1']))
    return winner['candidate'], summaries


def checkpoint_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completed_selection(path, config, split):
    required = ('selection.json', 'config.json', 'split.json', 'history.json', 'checkpoint.pt')
    if not all((path / name).is_file() for name in required) or (path / 'failure.json').exists():
        return None
    try:
        result = json.loads((path / 'selection.json').read_text())
        history = json.loads((path / 'history.json').read_text())
        epoch = result['best_checkpoint_epoch']
        if (result['status'] == 'validation_completed' and result['config'] == config
                and result['seed'] == config['seeds'][0] and result['subject_id'] == config['subject_id']
                and type(epoch) is int and 1 <= epoch <= len(history)
                and history[epoch - 1]['epoch'] == epoch
                and history[epoch - 1]['validation'] == result['validation']
                and json.loads((path / 'split.json').read_text()) == split
                and result['checkpoint_sha256'] == checkpoint_hash(path / 'checkpoint.pt')
                and json.loads((path / 'config.json').read_text()) == config):
            return result
    except (KeyError, IndexError, TypeError, ValueError, OSError):
        pass
    return None


def run_search(configs, *, restart=False):
    from agfl.datasets import load_dataset, get_split
    from agfl.engine import run_training, evaluate_saved_checkpoint, completed_run
    from agfl.reproducibility import provenance, seed_everything
    from agfl.storage import reset_run_artifacts, run_directory, write_json

    first = next(iter(configs.values()))[0]
    root = Path(first['output_dir'])
    current_provenance = provenance()
    # Serialize the whole search: a second launcher must not replace a chosen
    # candidate while this process is preparing its final test evaluation.
    with run_directory(root):
        # Fixed run locations are safe to resume only within the same search.
        # In particular, a smaller subject list must not retain old subjects in
        # selected/ and silently contaminate the new analysis.
        plan = {'candidate_order': list(configs), 'configurations': configs}
        plan_path = root / 'search_plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError('This output directory belongs to different search settings. Use a new --output-dir; identical launches resume automatically.')
        write_json(plan_path, plan)
        # Do not leave an earlier completion summary visible during a restart.
        for name in ('selection_report.json', 'search_result.json', 'search_report.md'):
            (root / name).unlink(missing_ok=True)
        grouped = defaultdict(dict)
        for name, experiments in configs.items():
            for config in experiments:
                grouped[config['subject_id']][name] = config
        choices, selected, all_results = {}, {}, []
        for subject, variants in sorted(grouped.items()):
            bundles, anchors, runs, prepared = {}, {}, {}, {}
            for name, config in variants.items():
                data_key = digest(config['data'])
                if data_key not in bundles:
                    bundles[data_key] = load_dataset('eeg', config['data'])
                bundle = bundles[data_key]
                if bundle.metadata['num_classes'] != 4 or set(bundle.groups) != {subject}:
                    raise ValueError('This search requires four-class BCI IV 2a for one subject at a time')
                runs[name], prepared[name] = [], []
                for seed in config['seeds']:
                    seed_everything(seed, config['deterministic'], config['threads'])
                    split = get_split(bundle, config['split'], seed, config['split_dir'])
                    anchor = (split['sample_ids'], split['class_counts'])
                    if seed in anchors and anchors[seed] != anchor:
                        raise ValueError('Candidates must use identical trial IDs/labels in every partition')
                    anchors[seed] = deepcopy(anchor)
                    resolved = deepcopy(config)
                    resolved.update(seeds=[seed], dataset_fingerprint=bundle.fingerprint,
                                    expected_split_id=split['split_id'], resolved_metadata=bundle.metadata,
                                    provenance=current_provenance)
                    path = root / 'candidates' / name / subject / f'seed_{seed}'
                    with run_directory(path):
                        previous = None if restart else completed_selection(path, resolved, split)
                        if previous is None:
                            # Replacing ANY candidate can change this split's
                            # winner. Hide its prior selected result immediately,
                            # so an abort cannot leave it looking up-to-date.
                            target = root / 'selected' / 'eeg' / f'subject_{subject}' / f'seed_{seed}'
                            if target.exists():
                                with run_directory(target):
                                    reset_run_artifacts(target)
                            reset_run_artifacts(path)
                            write_json(path / 'config.json', resolved)
                            write_json(path / 'split.json', split)
                            print(f'Validation search: {subject} {name} seed={seed}', flush=True)
                            try:
                                previous = run_training(resolved, bundle, split, path, validation_only=True)
                                previous['checkpoint_sha256'] = checkpoint_hash(path / 'checkpoint.pt')
                                write_json(path / 'selection.json', previous)
                            except BaseException as error:
                                write_json(path / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
                                raise
                        runs[name].append(previous)
                        prepared[name].append((resolved, split, path))
            choices[subject], selected[subject] = {}, []
            for position, seed in enumerate(next(iter(variants.values()))['seeds']):
                winner, summary = select_candidate({name: [records[position]] for name, records in runs.items()})
                choices[subject][str(seed)] = {'winner': winner, 'candidates': summary}
                selected[subject].append((winner, *prepared[winner][position]))
            # Release subjects' recording arrays before moving on. Final testing
            # starts only after every requested subject's choice is recorded.
            del bundles
        write_json(root / 'selection_report.json', {
            'status': 'settings_selected', 'choices': choices,
            'selection_metric': 'validation accuracy within each subject/seed split; macro F1 breaks ties',
            'provenance': current_provenance,
            'note': 'No test metrics entered candidate or checkpoint selection.'})
        for subject, selections in sorted(selected.items()):
            bundles = {}
            for winner, config, split, source in selections:
                data_key = digest(config['data'])
                if data_key not in bundles:
                    bundles[data_key] = load_dataset('eeg', config['data'])
                bundle = bundles[data_key]
                target = root / 'selected' / 'eeg' / f'subject_{subject}' / f"seed_{config['seeds'][0]}"
                with run_directory(target):
                    previous = None if restart else completed_run(target, config)
                    source_hash = checkpoint_hash(source / 'checkpoint.pt')
                    if (previous is not None and previous.get('search_checkpoint_sha256') == source_hash
                            and checkpoint_hash(target / 'checkpoint.pt') == source_hash):
                        all_results.append(previous)
                        continue
                    reset_run_artifacts(target)
                    for name in ('config.json', 'split.json', 'history.json', 'checkpoint.pt'):
                        shutil.copyfile(source / name, target / name)
                    selection = json.loads((source / 'selection.json').read_text())
                    try:
                        result = evaluate_saved_checkpoint(config, bundle, split, target,
                                                           elapsed_seconds=selection['elapsed_seconds'])
                        result.update(search_candidate=winner, search_checkpoint_sha256=source_hash)
                        write_json(target / 'result.json', result)
                        all_results.append(result)
                    except BaseException as error:
                        write_json(target / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
                        raise
        subject_scores = {subject: [r['test']['accuracy'] for r in all_results if r['subject_id'] == subject]
                          for subject in sorted(selected)}
        subject_means = {subject: statistics.mean(scores) for subject, scores in subject_scores.items()}
        complete = set(subject_means) == {f'A{i:02d}' for i in range(1, 10)}
        report = {
            'subjects': subject_means, 'n_subjects': len(subject_means), 'n_runs': len(all_results),
            'subject_seed_sd': {subject: statistics.stdev(scores) if len(scores) > 1 else None
                                for subject, scores in subject_scores.items()},
            'seeds': first['seeds'], 'complete_nine_subjects': complete,
            'mean_test_accuracy': statistics.mean(subject_means.values()),
            'between_subject_sd': statistics.stdev(subject_means.values()) if len(subject_means) > 1 else None,
            'target_accuracy': .75, 'target_met': complete and statistics.mean(subject_means.values()) >= .75,
            'choices': {s: {seed: choice['winner'] for seed, choice in by_seed.items()} for s, by_seed in choices.items()},
            'caveat': '75% means the mean across all nine subjects, not every subject. This measures a validation-selected tuning procedure, not one fixed hyperparameter setting. Seeds use overlapping random T-session trial splits and are not independent test cohorts. Previously inspected test scores are not a pristine new confirmation set.',
        }
        write_json(root / 'search_result.json', report)
        lines = ['# EEGNet + AGFL / BCI IV 2a', '',
                 f"Mean selected test accuracy: **{report['mean_test_accuracy']:.2%}** across {len(subject_means)}/9 subjects ({len(all_results)} runs).",
                 '', 'Settings were selected independently on validation within each subject/seed split.',
                 report['caveat'], '', '| Subject | Mean test accuracy | SD across seeds |',
                 '|---|---:|---:|']
        for subject, score in subject_means.items():
            sd = report['subject_seed_sd'][subject]
            lines.append(f"| {subject} | {score:.2%} | {sd:.2%} |" if sd is not None else f"| {subject} | {score:.2%} | — |")
        lines += ['', '| Subject | Seed | Selected candidate | Validation accuracy | Test accuracy | Checkpoint epoch |',
                  '|---|---:|---|---:|---:|---:|']
        for result in all_results:
            lines.append(f"| {result['subject_id']} | {result['seed']} | {result['search_candidate']} | {result['validation']['accuracy']:.2%} | {result['test']['accuracy']:.2%} | {result['best_checkpoint_epoch']} |")
        lines += ['', 'Candidate validation scores: `selection_report.json`. Overall selected metrics: `search_result.json`.',
                  'Run `analyze` on `selected/` to generate result figures. See the README for checkpoint diagnostic plots.', '']
        (root / 'search_report.md').write_text('\n'.join(lines))
        print(f"Selected EEGNet + AGFL: {report['mean_test_accuracy']:.2%} test accuracy across {len(subject_means)}/9 subjects", flush=True)
        return report
