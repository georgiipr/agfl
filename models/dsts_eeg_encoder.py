import torch
import torch.nn as nn
import torch.nn.functional as F

from depricated.agfl_layer_0 import AGFL
from standard_attention import StandardAttention

class DownProjector(nn.Module):
    def __init__(self, num_channels=22, dropout=0.3):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(64, 1)),
            nn.BatchNorm2d(16),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, num_channels)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(7, 1), stride=(7, 1)),
            nn.Dropout(dropout),
        )
        self.layer3 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 7, 8)),
            nn.Conv2d(32, 32, kernel_size=(16, 1), padding=0),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(7, 1), stride=(7, 1)),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        w, v = self.proj(x).chunk(2, dim=-1)
        return self.out(F.silu(w) * v)


class StableTransformerBlock(nn.Module):
    def __init__(self, dim=32, num_heads=2, mlp_ratio=4.0, attention_type='agfl', **attn_kwargs):
        super().__init__()
        self.attention_type = attention_type.lower()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        
        if self.attention_type == 'agfl':
            self.attn = AGFL(dim=dim, heads=num_heads, **attn_kwargs)
        elif self.attention_type == 'standard':
            self.attn = StandardAttention(dim=dim, heads=num_heads)
        else:
            self.attn = None
            
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))

    def forward(self, x):
        attn_in = self.norm1(x)
        
        if self.attn is not None:
            if self.attention_type == 'agfl':
                attn_out = self.attn(attn_in, layer_idx=0, L=1)
            else:
                attn_out = self.attn(attn_in)
            x = x + attn_out
            
        x = x + self.mlp(self.norm2(x))
        return x


class StableTransformer(nn.Module):
    def __init__(self, dim=32, num_heads=2, depth=4, mlp_ratio=4.0, attention_type='agfl', **attn_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [StableTransformerBlock(dim, num_heads, mlp_ratio, attention_type, **attn_kwargs) for _ in range(depth)]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


def _tcn_layer(channels, kernel_size, dilation, activation, dropout):
    return nn.Sequential(
        CausalConv1d(channels, channels, kernel_size, dilation),
        nn.BatchNorm1d(channels),
        activation,
        nn.Dropout(dropout),
    )


class TCN(nn.Module):
    def __init__(self, channels=32, kernel_size=3, dropout=0.3):
        super().__init__()
        self.stage1 = nn.Sequential(
            _tcn_layer(channels, kernel_size, dilation=1, activation=nn.SiLU(), dropout=dropout),
            _tcn_layer(channels, kernel_size, dilation=1, activation=nn.SiLU(), dropout=dropout),
        )
        self.stage2 = nn.Sequential(
            _tcn_layer(channels, kernel_size, dilation=2, activation=nn.ReLU(), dropout=dropout),
            _tcn_layer(channels, kernel_size, dilation=2, activation=nn.ReLU(), dropout=dropout),
        )
        self.final_act = nn.ReLU()

    def forward(self, x):
        x = self.stage1(x)
        x = self.final_act(self.stage2(x) + x)
        return x.transpose(1, 2)


class DSTS(nn.Module):
    def __init__(
        self,
        in_channels=32,
        dim=32,
        num_classes=2,
        depth=4,
        num_heads=2,
        dropout=0.3,
        attention_type='agfl',
        **attn_kwargs,
    ):
        super().__init__()
        self.tcn = TCN(channels=in_channels, dropout=dropout)
        self.transformer = StableTransformer(dim=dim, num_heads=num_heads, depth=depth, attention_type=attention_type, **attn_kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, num_classes),
        )

    def forward(self, x):
        h_tcn = self.tcn(x)[:, -1, :]
        h_tf = self.transformer(x.transpose(1, 2))[:, -1, :]
        return self.mlp(h_tcn + h_tf)


class DSTSEEGEncoder(nn.Module):
    def __init__(
        self,
        attention_type='agfl',
        num_classes=2,
        num_channels=22,
        num_branches=5,
        dropout=0.3,
        **attn_kwargs,
    ):
        super().__init__()
        self.down_projector = DownProjector(num_channels=num_channels, dropout=dropout)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(dropout),
                    DSTS(
                        in_channels=32,
                        dim=32,
                        num_classes=num_classes,
                        dropout=dropout,
                        attention_type=attention_type,
                        **attn_kwargs,
                    ),
                )
                for _ in range(num_branches)
            ]
        )

    @property
    def attn_blocks(self):
        return self.branches[0][1].transformer.blocks

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        if x.size(2) < x.size(3):
            x = x.transpose(2, 3) 
            
        x = self.down_projector(x)
        x = x.squeeze(-1)
        
        return torch.stack([branch(x) for branch in self.branches]).mean(dim=0)