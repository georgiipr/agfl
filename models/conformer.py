import torch
import torch.nn as nn
import torch.nn.functional as F

from agfl_layer import AGFL
from standard_none_attention import StandardAttention, NoAttention

class FeedForward(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class ConvModule(nn.Module):
    def __init__(self, dim, kernel_size=15):
        super().__init__()

        self.pointwise1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )
        self.batchnorm = nn.BatchNorm1d(dim)
        self.pointwise2 = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        # x: (B, N, D)
        x = x.transpose(1, 2)

        x = self.pointwise1(x)
        x = F.glu(x, dim=1)

        x = self.depthwise(x)
        x = self.batchnorm(x)
        x = F.silu(x)

        x = self.pointwise2(x)

        return x.transpose(1, 2)

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

class ConformerBlock(nn.Module):
    def __init__(self, dim, heads, K, mode="agfl", separate_W=True):
        super().__init__()

        self.mode = mode

        self.ff1 = FeedForward(dim)
        self.ff2 = FeedForward(dim)

        if mode == "agfl":
            self.attn = AGFL(dim, heads, K, separate_W)
        elif mode == "standard":
            self.attn = StandardAttention(dim, heads)
        else:
            self.attn = NoAttention(dim)

        self.conv = ConvModule(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

    def forward(self, x, layer_idx=None, L=None):
        x = x + 0.5 * self.ff1(self.norm1(x))
        if self.mode == "agfl":
            x = x + self.attn(self.norm2(x), layer_idx, L)
        else:
            x = x + self.attn(self.norm2(x))

        x = x + self.conv(self.norm3(x))
        x = x + 0.5 * self.ff2(self.norm4(x))

        return x

class ConformerModel(nn.Module):
    def __init__(
        self,
        dim=64,
        depth=3,
        heads=4,
        K=2,
        mode="agfl",
        separate_W=True
    ):
        super().__init__()
        self.mode = mode
        self.input = nn.Linear(1, dim)
        self.layers = nn.ModuleList([
            ConformerBlock(dim, heads, K, mode, separate_W)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.cls = nn.Linear(dim, 2)
        self.pos_emb = nn.Parameter(torch.randn(1, 256, dim))

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input(x)
        x = x + 0.1 * self.pos_emb[:, :x.size(1)]
        for i, layer in enumerate(self.layers):

            if self.mode == "agfl":
                x = layer(x, i, len(self.layers))
            else:
                x = layer(x)

        x = self.norm(x)

        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1).values
        x_pooled = x_mean + x_max
        return self.cls(x_pooled)