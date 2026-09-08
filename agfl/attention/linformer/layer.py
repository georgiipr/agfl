"""Learned sequence projections E/F from Wang et al., arXiv:2006.04768."""
import torch
from torch import nn
from ..projections import ProjectedMixer

class Linformer(ProjectedMixer):
    def __init__(self, options, tokens):
        super().__init__(options['dim'], options['heads'])
        rank = options['projection_rank']
        if type(rank) is not int or rank < 1:
            raise ValueError('projection_rank must be positive')
        self.tokens = tokens
        rank = min(rank, tokens)
        self.E = nn.Parameter(torch.empty(self.heads, rank, tokens))
        self.F = nn.Parameter(torch.empty(self.heads, rank, tokens))
        nn.init.normal_(self.E, std=tokens**-.5)
        nn.init.normal_(self.F, std=tokens**-.5)

    def forward(self, x):
        if x.shape[1] != self.tokens:
            raise ValueError('Linformer requires the configured fixed token count')
        q, k, v = self.project(x)
        k = torch.einsum('hrn,bhnd->bhrd', self.E, k)
        v = torch.einsum('hrn,bhnd->bhrd', self.F, v)
        weights = (q @ k.transpose(-1, -2) / self.head_dim**.5).softmax(-1)
        return self.combine(weights @ v)
