import torch
import torch.nn as nn

class StandardAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=heads, 
            dropout=dropout,
            batch_first=True
        )
        self.last_attn = None

    def forward(self, x):
        out, attn_weights = self.attn(x, x, x, need_weights=True)
        self.last_attn = attn_weights.detach() 
        return out

class NoAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ff = nn.Linear(dim, dim)

    def forward(self, x):
        return self.ff(x)