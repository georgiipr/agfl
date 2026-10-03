"""Checkpoint evidence for the per-electrode output gates of AGFL and MHA."""
import numpy as np

from agfl_speech.storage import write_json
from .common import slug


def output_gates(module):
    """Return per-head gates without pretending MHA has polynomial filters."""
    if getattr(module, 'attention_key', None) == 'mha':
        return list(module.output_gates) or [None] * module.heads
    return [f.output_gate for f in module.filters]


class TemporalGateCapture:
    def __init__(self, model):
        self.layers, self.hooks, self.sample_ids, self.unavailable = {}, [], None, []
        for name, module in model.named_modules():
            key = getattr(module, 'attention_key', None)
            is_agfl = key == 'agfl' and module.options.get('coefficient_conditioning') == 'token_contrast'
            is_mha = key == 'mha' and any(module.options.get(k, False) for k in ('temporal_bias', 'output_gate'))
            if not (is_agfl or is_mha):
                continue
            if getattr(model, 'pre_spatial', False):
                # One graph per time step: rows are trials x samples, not trials.
                self.unavailable.append(name)
                continue
            self.layers[name] = {'module': module, 'ids': [], 'graphs': [], 'input_norm': [],
                                 'output_norm': [], 'gates': {h: [] for h in range(module.heads)}}
            self.hooks.append(module.register_forward_hook(self._attention_hook(name)))
            for head, gate in enumerate(output_gates(module)):
                if gate is not None:
                    self.hooks.append(gate.register_forward_hook(self._gate_hook(name, head)))

    def start_batch(self, sample_ids):
        self.sample_ids = list(sample_ids)

    def _gate_hook(self, name, head):
        def hook(module, args, output):
            import torch
            with torch.no_grad():
                gates = module.gate_values(args[1].detach(), args[2].detach())
            self.layers[name]['gates'][head].append(gates.squeeze(-1).float().cpu().numpy().copy())
        return hook

    def _attention_hook(self, name):
        def hook(module, args, output):
            import torch
            if self.sample_ids is None or len(args[0]) != len(self.sample_ids):
                raise ValueError('Temporal/gate diagnostics require aligned sample IDs')
            state = self.layers[name]
            state['ids'].extend(self.sample_ids)
            graph = module.last_attn if module.attention_key == 'mha' else module.last_adj.permute(1, 0, 2, 3)
            state['graphs'].append(graph.float().cpu().numpy().copy())
            state['input_norm'].append(torch.linalg.vector_norm(args[0].detach().float(), dim=(1, 2)).cpu().numpy())
            state['output_norm'].append(torch.linalg.vector_norm(output.detach().float(), dim=(1, 2)).cpu().numpy())
            for head, gate in enumerate(output_gates(module)):
                if gate is None:
                    state['gates'][head].append(np.ones(args[0].shape[:2], dtype=np.float32))
        return hook


def temporal_gate_figures(writer, capture, sample_ids, labels, partition, *, metadata=None):
    import torch
    summaries = {}
    for name, state in capture.layers.items():
        if state['ids'] != list(sample_ids):
            raise ValueError('Temporal/gate capture does not match selected diagnostic IDs')
        module = state['module']
        graphs = np.concatenate(state['graphs'])
        gates = np.stack([np.concatenate(state['gates'][h]) for h in range(module.heads)], axis=1)
        input_norm, output_norm = np.concatenate(state['input_norm']), np.concatenate(state['output_norm'])
        tokens = graphs.shape[-1]
        electrode = getattr(module, 'token_axis', None) == 'electrode'
        folder = 'electrode_gate' if electrode else 'temporal_gate'
        node_labels = ((metadata or {}).get('channel_names') or
                       [f'Electrode {i + 1}' for i in range(tokens)]) if electrode else [str(i) for i in range(tokens)]
        if len(node_labels) != tokens:
            raise ValueError('Gate labels must match the attention node count')
        with torch.no_grad():
            bias = (module.temporal_bias(tokens).float().cpu().numpy() if module.temporal_bias is not None
                    else np.zeros((module.heads, tokens, tokens), dtype=np.float32))
            weights = (module.temporal_bias.weight.detach().float().cpu().numpy() if module.temporal_bias is not None
                       else np.zeros((module.heads, 3), dtype=np.float32))
        valid = input_norm > 1e-12
        ratio = np.divide(output_norm, input_norm, out=np.zeros_like(input_norm), where=valid)
        if not all(np.isfinite(a).all() for a in (graphs, gates, bias, weights, input_norm, output_norm, ratio)):
            raise ValueError('Nonfinite temporal/gate diagnostic values')
        prefix = folder + '/' + slug(name)
        (writer.root / folder).mkdir(exist_ok=True)
        np.savez_compressed(writer.root / (prefix + '.npz'), sample_ids=np.asarray(sample_ids), labels=labels,
                            token_axis=np.asarray(module.token_axis), node_labels=np.asarray(node_labels, dtype=str),
                            graphs=graphs, gates=gates, temporal_bias=bias, temporal_weights=weights,
                            input_norm=input_norm, output_norm=output_norm, output_input_ratio=ratio,
                            output_input_ratio_valid=valid)
        summary = {'partition': partition, 'samples': len(sample_ids), 'arrays': prefix + '.npz',
                   'token_axis': getattr(module, 'token_axis', 'unspecified'), 'node_labels': node_labels,
                   'temporal_bias_enabled': module.temporal_bias is not None,
                   'attention': module.attention_key,
                   'output_gate_enabled': any(g is not None for g in output_gates(module)),
                   'temporal_weight_columns': ['absolute_distance', 'signed_lag_source_ramp', 'self_edge'],
                   'temporal_weights': weights.tolist(), 'gate_mean_per_head': gates.mean(axis=(0, 2)).astype(float).tolist(),
                   'gate_std_per_head': gates.std(axis=(0, 2)).astype(float).tolist(),
                   'output_input_ratio_mean': float(ratio[valid].mean()) if valid.any() else None,
                   'output_input_ratio_valid_samples': int(valid.sum()),
                   'note': 'Gates multiply the AGFL hop mixture or MHA first-hop message before output projection. Graphs are pre-gate routing weights. Existing AGFL hop norms describe the pre-gate mixture. Z/X norms include output projection; the outer residual is unchanged.'}
        summaries[name] = summary
        if electrode:
            figure, axes = writer.plt.subplots(1, module.heads, figsize=(5 * module.heads, 4), squeeze=False, constrained_layout=True)
            for head, axis in enumerate(axes[0]):
                artist = axis.imshow(gates[:, head], aspect='auto', vmin=0, vmax=2, cmap='viridis')
                axis.set(title=f'Head {head}: output gate', xlabel='Electrode', ylabel=f'{partition} sample',
                         xticks=range(tokens), xticklabels=node_labels)
                axis.tick_params(axis='x', rotation=90, labelsize=6)
                figure.colorbar(artist, ax=axis)
            writer.save(figure, prefix, 'Per-electrode output gates on selected trials. Disabled gates equal one. '
                        'No temporal score bias is used on electrode indices; gate values are not causal importance.')
            continue
        figure, axes = writer.plt.subplots(2, module.heads, figsize=(4 * module.heads, 7), squeeze=False, constrained_layout=True)
        for head in range(module.heads):
            limit = max(float(np.abs(bias[head]).max()), 1e-6)
            artist = axes[0, head].imshow(bias[head], vmin=-limit, vmax=limit, cmap='coolwarm')
            axes[0, head].set(title=f'Head {head}: score bias', xlabel='Source token', ylabel='Destination token')
            figure.colorbar(artist, ax=axes[0, head])
            artist = axes[1, head].imshow(gates[:, head], aspect='auto', vmin=0, vmax=2, cmap='viridis')
            axes[1, head].set(title=f'Head {head}: output gate', xlabel='Token', ylabel=f'{partition} sample')
            figure.colorbar(artist, ax=axes[1, head])
        writer.save(figure, prefix, 'Learned temporal score biases and every selected sample/token gate. Disabled bias is zero; disabled gate is one. Values alone do not establish a benefit.')
    if summaries:
        electrode = all(s['token_axis'] == 'electrode' for s in summaries.values())
        write_json(writer.root / ('electrode_gate.json' if electrode else 'temporal_gate.json'), summaries)
    return summaries


def validation_ablation_figures(writer, model, inputs, sample_ids, labels, original_probabilities, device, partition):
    """Remove trained additions temporarily, without updating saved results.

    This is sensitivity at a trained checkpoint, not a retrained control. All
    parameters are restored even on failure. No test-set ablations are run.
    """
    import torch
    modules = [m for m in model.modules() if getattr(m, 'attention_key', None) in {'agfl', 'mha'}]
    parameters = {'temporal_bias': [], 'output_gate': []}
    for module in modules:
        if module.temporal_bias is not None:
            parameters['temporal_bias'].extend(module.temporal_bias.parameters())
        for gate in output_gates(module):
            if gate is not None:
                parameters['output_gate'].extend(gate.parameters())
    enabled = [name for name, values in parameters.items() if values]
    if not enabled or partition != 'validation':
        return None
    if model.training:
        raise ValueError('Diagnostic ablation requires eval mode')
    cases = [(name, [name]) for name in enabled]
    if len(enabled) == 2:
        cases.append(('both', enabled))
    arrays = {'sample_ids': np.asarray(sample_ids), 'labels': labels, 'original_probabilities': original_probabilities}
    original_correct = original_probabilities.argmax(-1) == labels
    summaries = []
    for name, removed in cases:
        selected = [p for key in removed for p in parameters[key]]
        saved = [p.detach().clone() for p in selected]
        try:
            with torch.no_grad():
                for parameter in selected:
                    parameter.zero_()
                predictions = [model(torch.from_numpy(inputs[start:start+16]).to(device)).softmax(-1).cpu().numpy()
                               for start in range(0, len(inputs), 16)]
        finally:
            with torch.no_grad():
                for parameter, value in zip(selected, saved):
                    parameter.copy_(value)
        probabilities = np.concatenate(predictions)
        if not np.isfinite(probabilities).all():
            raise ValueError('Nonfinite validation ablation probabilities')
        arrays['without_' + name] = probabilities
        correct = probabilities.argmax(-1) == labels
        summaries.append({'removed': name, 'accuracy': float(correct.mean()),
                          'accuracy_delta_from_original': float(correct.mean() - original_correct.mean()),
                          'fixed': int((correct & ~original_correct).sum()),
                          'lost': int((~correct & original_correct).sum()),
                          'changed_predictions': int((probabilities.argmax(-1) != original_probabilities.argmax(-1)).sum()),
                          'max_probability_change': float(np.abs(probabilities - original_probabilities).max())})
    folder_name = 'electrode_gate' if getattr(model, 'token_axis', None) == 'electrode' else 'temporal_gate'
    folder = writer.root / folder_name
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / 'validation_ablation.npz', **arrays)
    result = {'partition': partition, 'samples': len(labels), 'original_accuracy': float(original_correct.mean()),
              'ablations': summaries, 'arrays': folder_name + '/validation_ablation.npz',
              'report_file': folder_name + '/validation_ablation.json',
              'note': 'Checkpoint sensitivity only, using selected validation examples. No retraining, test selection, metric replacement or permanent parameter changes.'}
    write_json(folder / 'validation_ablation.json', result)
    figure, axis = writer.plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.bar(['Original'] + ['Without ' + row['removed'] for row in summaries],
             [100 * result['original_accuracy']] + [100 * row['accuracy'] for row in summaries])
    axis.set(ylabel='Selected validation accuracy (%)', title='Sensitivity to disabling trained additions', ylim=(0, 100))
    writer.save(figure, folder_name + '/validation_ablation', result['note'])
    return result
