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
        # Form probabilities before reducing precision: retained small weights
        # can otherwise underflow before post-softmax renormalization.
        working = scores.float() if scores.dtype in {torch.float16, torch.bfloat16} else scores
        graph = working.softmax(-1) * mask
        if renormalize:
            graph = graph / graph.sum(-1, keepdim=True).clamp_min(1e-12)
        return graph.to(scores.dtype)
    if method == 'none':
        return scores * mask
    # In float16, 1e-12 rounds to zero. Keep the activation and normalization
    # in float32 so an empty row has finite values and a finite ReLU gradient.
    # Float32/64 formulas and the historical score-stage softmax stay unchanged.
    working = scores.float() if scores.dtype in {torch.float16, torch.bfloat16} else scores
    graph = F.relu(working) * mask
    if method == 'row':
        graph = graph / graph.sum(-1, keepdim=True).clamp_min(1e-12)
        return graph.to(scores.dtype)
    if method == 'symmetric':
        graph = (graph + graph.transpose(-1, -2)) / 2
        inv = graph.sum(-1).clamp_min(1e-12).rsqrt()
        graph = inv.unsqueeze(-1) * graph * inv.unsqueeze(-2)
        return graph.to(scores.dtype)
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
    def __init__(self, dim, scaling='temperature', temperature_init=1.0):
        super().__init__()
        if (isinstance(temperature_init, bool) or not isinstance(temperature_init, (int, float))
                or not 0.1 <= temperature_init <= 5.0 or not math.isfinite(temperature_init)):
            raise ValueError('temperature_init must be a finite number in [0.1, 5.0]')
        if scaling != 'temperature' and temperature_init != 1.0:
            raise ValueError('Nondefault temperature_init requires score_scaling=temperature')
        # Retain the scalar state-dict entry and consume no random draws. Old
        # checkpoints overwrite this initialization with their learned value.
        self.temperature = nn.Parameter(torch.tensor(float(temperature_init)),
                                        requires_grad=scaling == 'temperature')
        self.dim, self.scaling = dim, scaling

    def forward(self, q, k=None):
        raw = similarity_scores(q, q if k is None else k)
        return raw, scale_scores(raw, self.dim, self.temperature, self.scaling)


class _IdentityLinear(nn.Linear):
    """A Linear with deterministic initialization and unchanged state keys."""
    def reset_parameters(self):
        # Linear.__init__ calls this after allocating its parameters. Avoid its
        # random initialization so later Q/K/V/output weights keep their draws.
        nn.init.eye_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class _ZeroLinear(nn.Linear):
    """A learnable zero map without consuming initialization random draws."""
    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class GraphFilter(nn.Module):
    def __init__(self, dim, options):
        super().__init__()
        self.options = options
        if type(options['learnable_coefficients']) is not bool:
            raise ValueError('learnable_coefficients must be a boolean')
        self.conditioning = options.get('coefficient_conditioning', 'static')
        if self.conditioning not in {'static', 'trial_power', 'token_contrast'}:
            raise ValueError('coefficient_conditioning must be static, trial_power or token_contrast')
        self.conditioning_scale = options.get('coefficient_conditioning_scale', 0.5)
        if (isinstance(self.conditioning_scale, bool)
                or not isinstance(self.conditioning_scale, (int, float))
                or not math.isfinite(self.conditioning_scale) or self.conditioning_scale <= 0):
            raise ValueError('coefficient_conditioning_scale must be a finite positive number')
        if self.conditioning != 'static':
            if options['coefficient_activation'] != 'identity' or not options['learnable_coefficients']:
                raise ValueError(f'{self.conditioning} conditioning requires identity activation and learnable coefficients')
            if dim < 2:
                raise ValueError(f'{self.conditioning} conditioning requires at least two features per head')
        if self.conditioning == 'token_contrast' and options['K'] < 1:
            raise ValueError('token_contrast conditioning requires K >= 1')
        if options['coefficient_init'] == 'zeros':
            if options['coefficient_activation'] == 'relu':
                raise ValueError('Zero-initialized ReLU coefficients cannot learn; use a nonzero coefficient_init')
            if not options['learnable_coefficients'] and options['coefficient_activation'] == 'identity':
                raise ValueError('Frozen zero coefficients disable graph filtering; use a nonzero coefficient_init')
        taps = options['K'] + 1
        initial = torch.zeros(taps)
        if options['coefficient_init'] in {'one_hop', 'identity_one_hop'}:
            if taps < 2 or options['coefficient_activation'] != 'identity':
                raise ValueError(f"{options['coefficient_init']} initialization requires K >= 1 and identity coefficients")
            if options['coefficient_init'] == 'identity_one_hop':
                # Preserve a local value path from initialization without
                # increasing the sum of tap weights; higher taps still learn.
                initial[:2] = 0.5
            else:
                # With dense Q/K/V attention, sqrt(dim) scaling and value-only
                # taps, A @ V is exactly the MHA formula. Zero higher-order
                # taps remain learnable under identity activation.
                initial[1] = 1.
        elif options['coefficient_init'] != 'zeros':
            initial = torch.ones(taps) if options['coefficient_init'] == 'uniform' else torch.tensor([2.**-k for k in range(taps)])
            initial /= initial.sum()
            if options['coefficient_activation'] == 'softmax':
                initial = initial.log()
            elif options['coefficient_activation'] == 'sigmoid':
                initial = torch.logit(initial.clamp(.001, .999))
        self.alpha_logits = nn.Parameter(initial, requires_grad=options['learnable_coefficients'])
        self.projection = options['filter_projection']
        projection_init = options.get('filter_projection_init', 'pytorch')
        if projection_init not in ('pytorch', 'identity'):
            raise ValueError('filter_projection_init must be pytorch or identity')
        if projection_init == 'identity' and self.projection not in {'shared', 'separate'}:
            raise ValueError('Identity filter_projection_init requires shared or separate filter_projection')
        projection_layer = _IdentityLinear if projection_init == 'identity' else nn.Linear
        if self.projection == 'separate':
            self.W = nn.ModuleList([projection_layer(dim, dim, bias=False) for _ in range(taps)])
        elif self.projection == 'shared':
            self.W = projection_layer(dim, dim, bias=False)
        else:
            self.W = nn.Identity()
        # No bias: alpha_logits already supplies each hop's static offset.
        # The zero map preserves the selected static recipe at initialization,
        # including RNG state and the initialization of every later parameter.
        descriptor_dim = 2 * dim if self.conditioning == 'token_contrast' else dim
        self.coefficient_gate = (_ZeroLinear(descriptor_dim, taps, bias=False)
                                 if self.conditioning != 'static' else None)

    def _static_coefficients(self):
        activations = {'identity': lambda x: x, 'relu': F.relu, 'sigmoid': torch.sigmoid,
                       'softmax': lambda x: x.softmax(-1)}
        return activations[self.options['coefficient_activation']](self.alpha_logits)

    def coefficient_values(self, values, graph=None, *, first_hop=None):
        """Effective coefficients for these head-local value features.

        Static/trial-power coefficients have axes [batch, hop]; token-contrast
        coefficients have axes [batch, token, hop]. Token routing redistributes
        the hop mixture without changing its sum. Neither descriptor uses batch
        statistics, labels, caches or learnable normalization.
        """
        alpha = self._static_coefficients().unsqueeze(0).expand(values.shape[0], -1)
        if self.coefficient_gate is None:
            return alpha
        if self.conditioning == 'token_contrast':
            if first_hop is None:
                if graph is None:
                    raise ValueError('token_contrast coefficients require graph or first_hop')
                first_hop = tuple(propagate(graph, values, 1, self.options['agfl_variant'],
                                            self.options['hop_normalization']))[1]
            if first_hop.shape != values.shape:
                raise ValueError('token_contrast first_hop must match value dimensions')
            with torch.autocast(device_type=values.device.type, enabled=False):
                working = values if values.dtype == torch.float64 else values.float()
                # Promote separately: even finite half-precision operands can
                # overflow during subtraction before a later conversion.
                contrast = first_hop.to(working.dtype) - working
                descriptor = torch.cat((
                    F.layer_norm(working, (values.shape[-1],), eps=1e-5),
                    F.layer_norm(contrast, (values.shape[-1],), eps=1e-5)), dim=-1)
                routed = F.linear(descriptor, self.coefficient_gate.weight.to(working.dtype)).tanh()
                delta = (self.conditioning_scale / 2) * (routed - routed.mean(dim=-1, keepdim=True))
                coefficients = alpha.to(working.dtype).unsqueeze(1) + delta
            return coefficients.to(self.alpha_logits.dtype)
        # Squaring float16 values before promotion can overflow even when all
        # inputs are finite. Keep descriptor and gate arithmetic in float32
        # under AMP, while preserving float64 reference calculations.
        with torch.autocast(device_type=values.device.type, enabled=False):
            working = values if values.dtype == torch.float64 else values.float()
            log_power = (working.square().mean(dim=-2) + 1e-6).log()
            descriptor = F.layer_norm(log_power, (values.shape[-1],), eps=1e-5)
            gate = F.linear(descriptor, self.coefficient_gate.weight.to(working.dtype))
            coefficients = alpha.to(working.dtype) + self.conditioning_scale * gate.tanh()
        return coefficients.to(self.alpha_logits.dtype)

    def forward(self, graph, values):
        if self.conditioning == 'token_contrast':
            # Reuse exactly the propagated features being mixed. No duplicate
            # graph construction, extra graph hop, or retained autograd cache.
            hops = tuple(propagate(graph, values, self.options['K'], self.options['agfl_variant'],
                                   self.options['hop_normalization']))
            alpha = self.coefficient_values(values, first_hop=hops[1])
            out = None
            for hop, propagated in enumerate(hops):
                projected = self.W[hop](propagated) if self.projection == 'separate' else self.W(propagated)
                term = alpha[:, :, hop, None].to(projected.dtype) * projected
                out = term if out is None else out + term
            return out
        # Preserve the historical scalar-per-hop arithmetic for static runs.
        alpha = self._static_coefficients() if self.coefficient_gate is None else self.coefficient_values(values)
        out = None
        for hop, propagated in enumerate(propagate(graph, values, self.options['K'], self.options['agfl_variant'],
                                                  self.options['hop_normalization'])):
            projected = self.W[hop](propagated) if self.projection == 'separate' else self.W(propagated)
            coefficient = alpha[hop] if self.coefficient_gate is None else alpha[:, hop, None, None].to(projected.dtype)
            term = coefficient * projected
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
            'coefficient_init': {'uniform', 'lower_order', 'zeros', 'one_hop', 'identity_one_hop'},
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
        self.builders = nn.ModuleList([
            GraphConstructor(self.dim_h, options['score_scaling'], options.get('temperature_init', 1.0))
            for _ in range(heads)
        ])
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
