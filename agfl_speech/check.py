"""Pre-flight verification on the experiment machine; nothing is trained or saved.

``python -m agfl_speech check`` loads the SI_Hom archive (which verifies it
against its manifest), builds every split protocol for the pooled cohort and for
each individual subject, and runs one forward and one backward pass of every
backbone/attention pair on real trials. A failure here is cheaper than a failure
after a queued training job has started.
"""
import tempfile

import numpy as np


def _split_summary(bundle, split):
    counts = {name: len(split[name]) for name in ('train', 'validation', 'test')}
    smallest = {name: int(min(split['class_counts'][name])) for name in counts}
    return (f"train/validation/test = {counts['train']}/{counts['validation']}/{counts['test']}, "
            f"smallest class count = {smallest['train']}/{smallest['validation']}/{smallest['test']}")


def run_check(data_dir=None, device='cpu'):
    import torch
    from torch.nn import functional as F

    from .attention import available_attentions
    from .config import resolve_config
    from .datasets import get_split, load_dataset
    from .datasets.si_hom import DATASET_KEY, EXPECTED_RELEASE, SUBJECTS, resolve_data_dir, subject_name
    from .datasets.splits import PROTOCOLS
    from .models import available_models, get_model_spec

    data = {} if data_dir is None else {'data_dir': data_dir}
    bundle = load_dataset(DATASET_KEY, data)
    metadata = bundle.metadata
    print(f"Data folder: {resolve_data_dir(metadata['preprocessing']['data_dir'])}")
    print(f"Archive verified against its manifest: {metadata['sources'][0]['name']} "
          f"sha256={metadata['sources'][0]['sha256'][:16]}...")
    print(f"Trials {len(bundle)}, electrodes {metadata['channels']}, samples {metadata['samples']} "
          f"at {metadata['sampling_rate']:g} Hz, classes {metadata['num_classes']}")
    print(f"Electrodes: {', '.join(metadata['channel_names'])}")
    print(f"Trials per subject: {metadata['subject_trials']}")
    print(f"Trials per class {metadata['label_names']}: {metadata['class_counts']}")
    print(f"Signal mean {float(bundle.x.mean()):.4f}, standard deviation {float(bundle.x.std()):.4f} "
          "(the authors z-scored every trial and channel)")
    print(f"Dataset fingerprint: {bundle.fingerprint}")
    observed = {'trials': len(bundle), 'samples': metadata['samples'],
                'author_test_trials': int(sum(metadata['author_test'])),
                'subject_trials': metadata['subject_trials']}
    if observed != EXPECTED_RELEASE:
        raise ValueError(f'The archive is not the released SI_Hom recordings: found {observed}, '
                         f'expected {EXPECTED_RELEASE}')
    print('Trial, sample and subject counts match the released recordings.')

    fractions = {'train': 0.6, 'validation': 0.2, 'test': 0.2}
    with tempfile.TemporaryDirectory() as directory:
        print('\nPooled cohort splits (seed 0):')
        for protocol in PROTOCOLS:
            split = get_split(bundle, {'protocol': protocol, **fractions}, 0, directory)
            print(f"  {protocol:13s} {_split_summary(bundle, split)}; test subjects {split['groups']['test']}")
        print('\nIndividual subject splits (seed 0):')
        for subject in SUBJECTS:
            single = load_dataset(DATASET_KEY, {**data, 'subjects': [subject]})
            for protocol in PROTOCOLS:
                if protocol == 'group':
                    continue
                split = get_split(single, {'protocol': protocol, **fractions}, 0, directory)
                print(f"  {subject_name(subject)} {protocol:13s} {_split_summary(single, split)}")

    target = torch.device(device)
    if target.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    # Two trials of every class, so BatchNorm sees a real batch.
    chosen = np.concatenate([np.flatnonzero(bundle.y == label)[:2] for label in range(metadata['num_classes'])])
    inputs = torch.from_numpy(bundle.x[chosen]).to(target)
    targets = torch.from_numpy(bundle.y[chosen]).to(target)
    print(f'\nBackbone/attention pairs on {len(chosen)} real trials ({target}):')
    variants = {'eegnet': [{'electrode_architecture': layout} for layout in ('spatial_fusion', 'pre_spatial', 'compact')]}
    for model_key in available_models():
        for attention_key in available_attentions():
            for options in variants.get(model_key, [{}]):
                config = resolve_config({'model': model_key, 'attention': attention_key, 'device': str(target),
                                         'data': data, 'model_options': options})
                torch.manual_seed(0)
                model = get_model_spec(model_key).build(
                    config['model_options'], metadata, attention_key, config['attention_options']).to(target)
                model.eval()
                with torch.no_grad():
                    logits = model(inputs)
                if tuple(logits.shape) != (len(chosen), metadata['num_classes']) or not bool(torch.isfinite(logits).all()):
                    raise ValueError(f'{model_key}/{attention_key}: invalid logits {tuple(logits.shape)}')
                model.train()
                loss = F.cross_entropy(model(inputs), targets)
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                    raise ValueError(f'{model_key}/{attention_key}: missing or non-finite gradients')
                layers = [m for m in model.modules() if getattr(m, 'is_attention', False)]
                label = model_key + (f" [{options['electrode_architecture']}]" if options else '')
                width = getattr(layers[0], 'options', {}).get('dim', '?')
                print(f"  {label:28s} {attention_key:6s} parameters={sum(p.numel() for p in model.parameters()):7d} "
                      f"attention layers={len(layers)} nodes={layers[0].num_tokens} token width={width} "
                      f"loss={float(loss):.3f}")
    print('\nAll checks passed. Nothing was trained or saved.')
