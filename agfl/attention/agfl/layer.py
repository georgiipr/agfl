"""AGFL with the active historical polynomial as its default.

The original state-dict layout and per-head initialization order are retained.
Alternative formulas and graph policies are explicit configuration ablations.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def sparsity_schedule(layer_idx, depth, smax=0.2, smin=0.8, alpha=3.0):
    return smin + (smax - smin) * math.exp(-alpha * layer_idx / depth)


def similarity_scores(q, k):
    return q @ k.transpose(-1, -2)


def scale_scores(scores, head_dim, temperature, scaling):
    if scaling == 'raw':
        return scores
    scale = math.sqrt(head_dim)
    if scaling == 'temperature':
        scale = scale * temperature.clamp(0.1, 5.0)
    return scores / scale


def retained_count(top_k, tokens, layer_idx, depth, schedule):
    if top_k is None:
        return tokens
    if top_k == 'scheduled':
        sparsity = sparsity_schedule(layer_idx, depth, **schedule)
        return max(1, int((1 - sparsity) * tokens))
    if type(top_k) is int and top_k > 0:
        return min(top_k, tokens)
    if type(top_k) is float and 0 < top_k <= 1:
        return max(1, int(top_k * tokens))
    raise ValueError('top_k must be scheduled, null, a positive integer, or a ratio in (0,1]')


def topk_mask(scores, count, ties='threshold'):
    if count >= scores.shape[-1]:
        return torch.ones_like(scores, dtype=torch.bool)
    if ties == 'threshold':
        # Preserve the original tie behavior: all scores equal to the kth survive.
        threshold = torch.topk(scores, count, dim=-1).values[..., -1:]
        return scores >= threshold
    indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :count]
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices, True)


def normalize_graph(scores, mask, method, stage, renormalize):
    if method == 'softmax':
        if stage in {'scores', 'raw_scores'}:
            return torch.softmax(scores.masked_fill(~mask, float('-inf')), dim=-1)
        graph = scores.softmax(-1) * mask
        return graph / graph.sum(-1, keepdim=True).clamp_min(1e-12) if renormalize else graph
    if method == 'none':
        return scores * mask
    graph = F.relu(scores) * mask
    if method == 'row':
        return graph / graph.sum(-1, keepdim=True).clamp_min(1e-12)
    if method == 'symmetric':
        graph = (graph + graph.transpose(-1, -2)) / 2
        inv = graph.sum(-1).clamp_min(1e-12).rsqrt()
        return inv.unsqueeze(-1) * graph * inv.unsqueeze(-2)
    raise ValueError(f'Unknown graph normalization {method}')


def propagate(graph, values, degree, variant, norm='feature', eps=1e-6):
    yield values
    current = values
    axes = -1 if norm == 'feature' else (-2, -1)
    reference = torch.linalg.vector_norm(values, dim=axes, keepdim=True) if variant == 'renormalized' else None
    for _ in range(degree):
        current = graph @ current
        if variant == 'renormalized':
            current = current * (reference / (torch.linalg.vector_norm(current, dim=axes, keepdim=True) + eps))
        yield current


class GraphConstructor(nn.Module):
    def __init__(self, dim, scaling='temperature'):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=scaling == 'temperature')
        self.dim, self.scaling = dim, scaling

    def forward(self, q, k=None):
        raw = similarity_scores(q, q if k is None else k)
        return raw, scale_scores(raw, self.dim, self.temperature, self.scaling)


class GraphFilter(nn.Module):
    def __init__(self, dim, options):
        super().__init__()
        self.options = options
        taps = options['K'] + 1
        initial = torch.zeros(taps)
        if options['coefficient_init'] != 'zeros':
            initial = torch.ones(taps) if options['coefficient_init'] == 'uniform' else torch.tensor([2.**-k for k in range(taps)])
            initial /= initial.sum()
            if options['coefficient_activation'] == 'softmax':
                initial = initial.log()
            elif options['coefficient_activation'] == 'sigmoid':
                initial = torch.logit(initial.clamp(.001, .999))
        self.alpha_logits = nn.Parameter(initial, requires_grad=options['learnable_coefficients'])
        self.projection = options['filter_projection']
        if self.projection == 'separate':
            self.W = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(taps)])
        elif self.projection == 'shared':
            self.W = nn.Linear(dim, dim, bias=False)
        else:
            self.W = nn.Identity()

    def forward(self, graph, values):
        activations = {'identity': lambda x: x, 'relu': F.relu, 'sigmoid': torch.sigmoid,
                       'softmax': lambda x: x.softmax(-1)}
        alpha = activations[self.options['coefficient_activation']](self.alpha_logits)
        out = None
        for hop, propagated in enumerate(propagate(graph, values, self.options['K'], self.options['agfl_variant'],
                                                  self.options['hop_normalization'])):
            projected = self.W[hop](propagated) if self.projection == 'separate' else self.W(propagated)
            term = alpha[hop] * projected
            out = term if out is None else out + term
        return out


class AGFL(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.options = options
        dim, heads = options['dim'], options['heads']
        if dim % heads:
            raise ValueError('dim must be divisible by heads')
        if type(options['K']) is not int or options['K'] < 0:
            raise ValueError('K must be a nonnegative maximum hop order')
        allowed = {
            'agfl_variant': {'polynomial', 'renormalized'},
            'top_k_stage': {'scores', 'raw_scores', 'softmax'},
            'top_k_ties': {'threshold', 'exact'},
            'graph_normalization': {'softmax', 'row', 'symmetric', 'none'},
            'score_scaling': {'sqrt_dim', 'temperature', 'raw'},
            'coefficient_activation': {'identity', 'relu', 'softmax', 'sigmoid'},
            'coefficient_init': {'uniform', 'lower_order', 'zeros'},
            'projection': {'qkv', 'split_input'},
            'filter_projection': {'none', 'shared', 'separate'},
            'hop_normalization': {'feature', 'frobenius'},
        }
        for key, choices in allowed.items():
            if options[key] not in choices:
                raise ValueError(f'{key} must be one of {sorted(choices)}')
        schedule = options['top_k_schedule']
        if set(schedule) != {'smax', 'smin', 'alpha'} or not 0 <= schedule['smax'] <= 1 or not 0 <= schedule['smin'] <= 1 or schedule['alpha'] < 0:
            raise ValueError('Invalid top_k_schedule')
        retained_count(options['top_k'], 10, 0, 1, schedule)
        if options['top_k_stage'] == 'softmax' and options['graph_normalization'] != 'softmax':
            raise ValueError('Post-softmax pruning requires graph_normalization=softmax')
        self.heads, self.dim_h = heads, dim // heads
        self.layer_idx, self.depth = 0, 1
        self.qkv = nn.Linear(dim, 3 * dim, bias=options['qkv_bias']) if options['projection'] == 'qkv' else None
        # This order exactly matches the historical default's initialization.
        self.builders = nn.ModuleList([GraphConstructor(self.dim_h, options['score_scaling']) for _ in range(heads)])
        self.filters = nn.ModuleList([GraphFilter(self.dim_h, options) for _ in range(heads)])
        self.proj = nn.Linear(dim, dim)
        self.last_adj = None

    def construct_graph(self, head, q, k, layer_idx, depth):
        raw, scores = self.builders[head](q, k)
        stage = self.options['top_k_stage']
        selection = raw if stage == 'raw_scores' else scores.softmax(-1) if stage == 'softmax' else scores
        count = retained_count(self.options['top_k'], scores.shape[-1], layer_idx, depth, self.options['top_k_schedule'])
        mask = topk_mask(selection, count, self.options['top_k_ties'])
        return normalize_graph(scores, mask, self.options['graph_normalization'], stage, self.options['post_topk_renormalize'])

    def forward(self, x, layer_idx=None, L=None):
        b, n, d = x.shape
        layer_idx = self.layer_idx if layer_idx is None else layer_idx
        depth = self.depth if L is None else L
        if not 0 <= layer_idx < depth:
            raise ValueError('Require 0 <= layer_idx < depth')
        if self.qkv is not None:
            q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.dim_h).permute(2, 0, 3, 1, 4).unbind(0)
        else:
            q = k = v = x.reshape(b, n, self.heads, self.dim_h).transpose(1, 2)
        outputs, graphs = [], []
        for head in range(self.heads):
            graph = self.construct_graph(head, q[:, head], k[:, head], layer_idx, depth)
            outputs.append(self.filters[head](graph, v[:, head]))
            graphs.append(graph.detach())
        # Match the historical [heads,batch,nodes,nodes] diagnostic layout.
        self.last_adj = torch.stack(graphs)
        return self.proj(torch.cat(outputs, dim=-1))
