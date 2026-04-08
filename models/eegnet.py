import torch
import torch.nn as nn
import torch.nn.functional as F

from agfl_layer import AGFL
from standard_none_attention import StandardAttention, NoAttention

class EEGNet(nn.Module):
    def __init__(self, num_channels=22, num_classes=2, samples=1000, 
                 F1=8, D=2, F2=16, attention_type='standard', dropout_rate=0.5):
        super(EEGNet, self).__init__()
        self.attention_type = attention_type
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        if self.attention_type == 'agfl':
            self.attn_blocks = nn.ModuleList([AGFL(dim=F2, heads=4, K=3, separate_W=True)])
        elif self.attention_type == 'standard':
            self.attn_blocks = nn.ModuleList([StandardAttention(dim=F2, heads=4)])
        else:
            self.attn_blocks = None


        out_features = self._calculate_out_features(samples, num_channels)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_features, num_classes)
        )

    def _calculate_out_features(self, samples, channels):
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, channels, samples)
            x = self.block1(dummy_x)
            x = self.block2(x)
            x = self.block3(x)
            return x.numel()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        if self.attn_blocks is not None:
            x = x.squeeze(2).transpose(1, 2)
            
            L = len(self.attn_blocks)
            for i, block in enumerate(self.attn_blocks):
                if self.attention_type == 'agfl':
                    x = block(x, layer_idx=i, L=L)
                else:
                    x = block(x)
                        
        out = self.classifier(x)
        return out