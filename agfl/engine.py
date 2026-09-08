"""Shared training policy, inherited by model-owned experiment definitions."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from .config import comparison_identity, experiment_identity
from .metrics import classification_metrics
from .reproducibility import seed_everything, seed_worker, provenance
from .storage import write_json, run_directory, reset_run_artifacts


class ClassificationLoss(nn.Module):
    def __init__(self, weights=None, kind='cross_entropy', gamma=2.0):
        super().__init__()
        self.register_buffer('weights', weights)
        self.kind, self.gamma = kind, gamma

    def forward(self, logits, targets):
        # Class weights multiply focal loss; they must not change p_t.
        ce = F.cross_entropy(logits, targets, reduction='none')
        loss = ce if self.kind == 'cross_entropy' else (1 - torch.exp(-ce)).pow(self.gamma) * ce
        if self.weights is not None:
            loss = loss * self.weights[targets]
        return loss.mean()


class Partition(Dataset):
    def __init__(self, bundle, indices, normalization=None, augmentation=None):
        self.bundle, self.indices = bundle, list(indices)
        self.normalization, self.augmentation = normalization, augmentation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        x = self.bundle.x[i].copy()
        if self.normalization:
            x = (x - self.normalization['mean']) / self.normalization['std']
        if self.augmentation:
            options = self.augmentation
            if options['shift']:
                shift = int(np.random.randint(-options['shift'], options['shift'] + 1))
                x = np.roll(x, shift, axis=-1)
                if shift > 0:
                    x[..., :shift] = 0
                elif shift < 0:
                    x[..., shift:] = 0
            if options['scale']:
                x *= np.random.uniform(1 - options['scale'], 1 + options['scale'])
            if options['noise']:
                x += np.random.normal(0, options['noise'], x.shape).astype(np.float32)
        return torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.tensor(self.bundle.y[i], dtype=torch.long)


def make_loaders(bundle, split, config, seed):
    from .datasets import validate_split
    validate_split(bundle, split)
    if split.get('diagnostic_only'):
        raise ValueError('Diagnostic historical splits cannot be used for research training')
    for name in ('train', 'validation', 'test'):
        if not split[name]:
            raise ValueError(f'Research experiments require a nonempty {name} partition')
    normalization = None
    if config['data'].get('normalization') == 'train_channel':
        x = bundle.x[split['train']]
        mean = x.mean(axis=(0, 2), dtype=np.float64)[:, None]
        std = x.std(axis=(0, 2), dtype=np.float64)[:, None]
        normalization = {'mean': mean.astype(np.float32), 'std': np.maximum(std, 1e-8).astype(np.float32)}
    options = config['training']
    if options['augmentation']['shift'] >= bundle.metadata['samples']:
        raise ValueError('Augmentation shift must be shorter than the signal window')
    loaders = {}
    for offset, name in enumerate(('train', 'validation', 'test')):
        partition = Partition(bundle, split[name], normalization, options['augmentation'] if name == 'train' else None)
        loaders[name] = DataLoader(
            partition, batch_size=options['batch_size'], shuffle=name == 'train',
            num_workers=options['num_workers'], worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed + offset),
            pin_memory=config['device'].startswith('cuda'), drop_last=False)
    return loaders, normalization


def make_optimizer(model, options):
    kwargs = {'lr': options['learning_rate'], 'weight_decay': options['weight_decay']}
    factories = {'adamw': torch.optim.AdamW, 'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
    if options['optimizer'] == 'sgd':
        kwargs['momentum'] = options['momentum']
    return factories[options['optimizer']](model.parameters(), **kwargs)


def make_scheduler(optimizer, options):
    if options['scheduler'] == 'none':
        return None
    if options['scheduler'] == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    epochs = options['epochs']
    warmup = min(options['warmup_epochs'], max(0, epochs - 1)) if options['scheduler'] == 'warmup_cosine' else 0
    minimum = options['min_lr_ratio']
    def factor(epoch):
        if warmup and epoch < warmup:
            return 0.1 + 0.9 * epoch / warmup
        progress = min(1., (epoch - warmup) / max(1, epochs - warmup))
        return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def evaluate(model, loader, device, loss_fn=None):
    model.eval()
    probabilities, targets, total_loss, total = [], [], 0., 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            if logits.ndim != 2 or logits.shape[0] != len(y):
                raise ValueError('Model must return unnormalized [batch, classes] logits')
            probabilities.append(logits.float().softmax(-1).cpu().numpy())
            targets.append(y.numpy())
            if loss_fn is not None:
                loss = float(loss_fn(logits, y.to(device)))
                if not math.isfinite(loss):
                    raise FloatingPointError('Nonfinite held-out loss')
                total_loss += loss * len(y)
            total += len(y)
    probabilities, targets = np.concatenate(probabilities), np.concatenate(targets)
    metrics = classification_metrics(targets, probabilities)
    metrics['loss'] = total_loss / total if loss_fn is not None else None
    return metrics, targets, probabilities


def run_training(config, bundle, split, run_dir):
    from .models import get_model_spec
    run_dir = Path(run_dir)
    seed = config['seeds'][0]
    seed_everything(seed, config['deterministic'], config['threads'])
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if config['training']['amp'] and device.type != 'cuda':
        raise ValueError('AMP is supported only for CUDA experiments')
    spec = get_model_spec(config['model'])
    model = spec.build(config['model_options'], bundle.metadata, config['attention'], config['attention_options']).to(device)
    loaders, normalization = make_loaders(bundle, split, config, seed)
    options = config['training']
    counts = np.bincount(bundle.y[split['train']], minlength=bundle.metadata['num_classes'])
    if np.any(counts == 0):
        raise ValueError(f'Training split lacks classes: {counts.tolist()}')
    weights = None
    if options['class_weights'] == 'balanced':
        weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=device)
    loss_fn = ClassificationLoss(weights, options['loss'], options['focal_gamma'])
    optimizer = make_optimizer(model, options)
    scheduler = make_scheduler(optimizer, options)
    scaler = torch.amp.GradScaler('cuda', enabled=options['amp'])
    history, best_value, best_epoch = [], None, None
    criterion = options['checkpoint_criterion']
    started = time.monotonic()
    for epoch in range(1, options['epochs'] + 1):
        model.train()
        loss_total, n, skipped_steps = 0., 0, 0
        learning_rate = optimizer.param_groups[0]['lr']
        for x, y in loaders['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=options['amp'], dtype=torch.float16):
                loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if options['gradient_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), options['gradient_clip'], error_if_nonfinite=not options['amp'])
            elif not options['amp'] and any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite gradient')
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped_steps += int(scaler.get_scale() < previous_scale)
            if hasattr(model, 'clip_weights'):
                model.clip_weights()
            loss_total += float(loss.detach()) * len(y)
            n += len(y)
        validation, _, _ = spec.evaluate(model, loaders['validation'], device, loss_fn)
        value = validation[criterion]
        if value is None or not math.isfinite(value):
            raise ValueError(f'Checkpoint criterion {criterion} is undefined on validation data')
        better = best_value is None or (value < best_value if criterion == 'loss' else value > best_value)
        if better:
            best_value, best_epoch = value, epoch
            torch.save({
                'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'epoch': epoch, 'validation': validation, 'config': config,
                'split_id': split['split_id'], 'dataset_fingerprint': bundle.fingerprint,
                'normalization': normalization}, run_dir / 'checkpoint.pt')
        history.append({'epoch': epoch, 'train_loss': loss_total / n,
                        'validation': validation, 'learning_rate': learning_rate,
                        'amp_skipped_steps': skipped_steps})
        if scheduler is not None:
            scheduler.step(validation['loss']) if options['scheduler'] == 'plateau' else scheduler.step()
        write_json(run_dir / 'history.json', history)
        print(f"{config['dataset']}/{config['model']}/{config['attention']} seed={seed} epoch={epoch}/{options['epochs']} "
              f"loss={loss_total/n:.4f} val_accuracy={validation['accuracy']:.4f}", flush=True)
    checkpoint = torch.load(run_dir / 'checkpoint.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    validation, val_y, val_prob = spec.evaluate(model, loaders['validation'], device, loss_fn)
    # First and only test evaluation. Test outputs never select a checkpoint.
    test, test_y, test_prob = spec.evaluate(model, loaders['test'], device, loss_fn)
    test_groups = bundle.groups[split['test']]
    group_metrics = {str(group): classification_metrics(test_y[test_groups == group], test_prob[test_groups == group])
                     for group in sorted(set(test_groups))}
    np.savez_compressed(run_dir / 'predictions.npz', validation_targets=val_y, validation_probabilities=val_prob,
                        test_targets=test_y, test_probabilities=test_prob,
                        validation_ids=np.asarray(bundle.sample_ids)[split['validation']],
                        test_ids=np.asarray(bundle.sample_ids)[split['test']])
    result = {
        'schema_version': 2, 'status': 'completed', 'dataset': config['dataset'],
        'model': config['model'], 'attention': config['attention'], 'model_variant': config['model_variant'], 'seed': seed,
        'split_id': split['split_id'], 'dataset_fingerprint': bundle.fingerprint,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'best_checkpoint_epoch': best_epoch, 'validation': validation, 'test': test,
        'test_by_subject': group_metrics,
        'training_class_counts': counts.tolist(), 'class_weights': None if weights is None else weights.cpu().tolist(),
        'elapsed_seconds': time.monotonic() - started, 'config': config,
        'experiment_id': experiment_identity(config), 'comparison_id': comparison_identity(config),
        'token_axis': model.token_axis,
        'num_tokens': getattr(model, 'num_tokens', None),
        'attention_modules': [{'name': name, 'attention': module.attention_key,
                               'num_tokens': module.num_tokens, 'token_axis': module.token_axis}
                              for name, module in model.named_modules() if getattr(module, 'is_attention', False)],
        'synthetic': bundle.metadata.get('synthetic', False)}
    write_json(run_dir / 'result.json', result)
    return result


def completed_run(run_dir, config):
    """Only fully written, matching runs are eligible for --skip-completed."""
    required = ('result.json', 'config.json', 'split.json', 'history.json', 'checkpoint.pt', 'predictions.npz')
    if not all((run_dir / name).is_file() for name in required):
        return None
    if (run_dir / 'failure.json').exists():
        return None
    try:
        previous = json.loads((run_dir / 'result.json').read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(previous, dict):
        return None
    expected = {
        'status': 'completed', 'seed': config['seeds'][0],
        'model': config['model'], 'attention': config['attention'], 'dataset': config['dataset'],
        'split_id': config['expected_split_id'], 'dataset_fingerprint': config['dataset_fingerprint'],
        'experiment_id': experiment_identity(config), 'comparison_id': comparison_identity(config),
    }
    return previous if all(previous.get(key) == value for key, value in expected.items()) else None


def run_experiment(config, skip_completed=False):
    from .config import resolve_config
    from .models import get_model_spec
    config = resolve_config(config)
    spec = get_model_spec(config['model'])
    seed_everything(config['seeds'][0], config['deterministic'], config['threads'])
    current_provenance = provenance()
    expected_provenance = config.get('provenance')
    if expected_provenance and (expected_provenance['source_sha256'] != current_provenance['source_sha256']
                                or expected_provenance['packages'] != current_provenance['packages']):
        raise ValueError('Saved configuration requires its recorded source and package versions; use a fresh experiment config to change implementation')
    results, bundle = [], None
    for seed in config['seeds']:
        seed_everything(seed, config['deterministic'], config['threads'])
        if bundle is None:
            bundle, split = spec.prepare_data(config, seed)
        else:
            from .datasets import get_split
            split = get_split(bundle, config['split'], seed, config['split_dir'])
        if config.get('dataset_fingerprint') not in (None, bundle.fingerprint):
            raise ValueError('Dataset bytes/preprocessing differ from the saved configuration')
        if config.get('expected_split_id') not in (None, split['split_id']):
            raise ValueError('Generated split differs from the saved configuration')
        resolved = copy.deepcopy(config)
        resolved.update(seeds=[seed], dataset_fingerprint=bundle.fingerprint,
                        expected_split_id=split['split_id'], resolved_metadata=bundle.metadata,
                        provenance=current_provenance)
        run_dir = Path(config['output_dir']) / config['dataset'] / f"{config['model']}-{config['attention']}-{experiment_identity(resolved)}" / f'seed_{seed}'
        with run_directory(run_dir):
            previous = completed_run(run_dir, resolved) if skip_completed else None
            if previous is not None:
                results.append(previous)
                continue
            if (run_dir / 'config.json').exists():
                print(f'Restarting {run_dir} from epoch 1; replacing previous run artifacts', flush=True)
            reset_run_artifacts(run_dir)
            write_json(run_dir / 'config.json', resolved)
            write_json(run_dir / 'split.json', split)
            try:
                results.append(spec.run(resolved, bundle, split, run_dir))
            except (Exception, KeyboardInterrupt) as error:
                write_json(run_dir / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
                raise
    return results
