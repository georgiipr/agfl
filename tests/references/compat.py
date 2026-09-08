"""RMSNorm formula for PyTorch 2.2 (which predates nn.RMSNorm)."""
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight

# Use the original model's native operator on the intended modern target stack.
RMSNorm = getattr(nn, 'RMSNorm', RMSNorm)
