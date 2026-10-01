"""Bounded diagnostic maps, evaluated only after an explicit diagnose request."""
import numpy as np
import torch


def mixer_map(mixer, inputs, max_tokens=256):
    """Return [B,H,N,N] graph/effective value-mixing weights and their meaning.

    AGFL returns its constructed graph, before polynomial filtering. Linformer
    and Nyström maps can be signed and must not be labeled probabilities.
    """
    from agfl.attention.agfl.layer import AGFL
    from agfl.attention.mha.layer import MultiHeadAttention
    from agfl.attention.performer.layer import Performer
    from agfl.attention.linformer.layer import Linformer
    from agfl.attention.nystromformer.layer import Nystromformer

    if inputs.shape[1] > max_tokens:
        return None, f'Map omitted: {inputs.shape[1]} tokens exceeds diagnostic limit {max_tokens}'
    if isinstance(mixer, AGFL):
        return mixer.last_adj.permute(1, 0, 2, 3).detach(), 'AGFL graph A before polynomial/renormalized filtering'
    if not isinstance(mixer, (MultiHeadAttention, Performer, Linformer, Nystromformer)):
        return None, f'No map adapter for {type(mixer).__name__}'
    q, k, _ = mixer.project(inputs)
    if isinstance(mixer, MultiHeadAttention):
        scores = q @ k.transpose(-1, -2) / mixer.head_dim**.5
        if mixer.temporal_bias is not None:
            scores = scores + mixer.temporal_bias(inputs.shape[1]).to(scores.dtype)
        kind = 'MHA routing probabilities before output gating'
        if mixer.temporal_bias is not None:
            kind += ', including learned temporal bias'
        return scores.softmax(-1), kind
    if isinstance(mixer, Performer):
        q, k = mixer.feature_map(q, True), mixer.feature_map(k, False)
        denominator = (q * k.sum(-2, keepdim=True)).sum(-1, keepdim=True).clamp_min(1e-12)
        return (q @ k.transpose(-1, -2)) / denominator, 'Performer effective random-feature weights'
    if isinstance(mixer, Linformer):
        compressed_keys = torch.einsum('hrn,bhnd->bhrd', mixer.E, k)
        weights = (q @ compressed_keys.transpose(-1, -2) / mixer.head_dim**.5).softmax(-1)
        return torch.einsum('bhnr,hrm->bhnm', weights, mixer.F), 'Linformer effective value-mixing weights (may be signed)'
    identity = torch.eye(inputs.shape[1], device=inputs.device, dtype=inputs.dtype)
    return mixer.mix(q, k, identity), 'Nystrom effective value-mixing weights (may be signed)'


def per_trial_maps(maps, trials):
    """Average trial-major per-time-step maps back to one [B,H,N,N] map per trial.

    EEGNet ``pre_spatial`` presents one electrode set per time step, so the
    attention batch axis is trials x time steps with row ``b * T + t``. Maps
    whose row count is not a multiple of ``trials`` are returned unchanged.
    """
    count = maps.shape[0]
    if trials < 1 or count == trials or count % trials:
        return maps
    return maps.reshape(trials, count // trials, *maps.shape[1:]).mean(1)


def map_statistics(maps, tolerance=1e-8):
    """Entropy only exists here for nonnegative, normalized, nonempty rows."""
    values = np.asarray(maps)
    if values.ndim != 4 or not np.isfinite(values).all():
        raise ValueError('Expected finite [B,H,N,N] mixer maps')
    rows = values.sum(-1)
    valid = (values >= 0).all(-1) & np.isclose(rows, 1, rtol=1e-5, atol=1e-7)
    entropy = -(values.clip(1e-30) * np.log(values.clip(1e-30))).sum(-1)
    return {
        'sparsity_per_head': (np.abs(values) <= tolerance).mean(axis=(0, 2, 3)).tolist(),
        'entropy_per_head': [float(entropy[:, head][valid[:, head]].mean()) if valid[:, head].all() else None
                             for head in range(values.shape[1])],
        'zero_tolerance': tolerance,
        'entropy_units': 'nats; undefined unless every row is nonnegative and sums to one',
    }
