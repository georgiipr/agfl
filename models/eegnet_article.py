import torch
import torch.nn as nn
import torch.nn.functional as F

from depricated.agfl_layer_0 import AGFL
from standard_attention import StandardAttention

class EEGNet(nn.Module):
    def __init__(self, num_channels=22, num_classes=4, time_points=1000, temp_kernel=32,
                 f1=16, d=2, f2=32, pk1=8, pk2=16, attention_type='standard', 
                 dropout_rate=0.5, max_norm1=1.0, max_norm2=0.25):
        super(EEGNet, self).__init__()
        
        self.attention_type = attention_type
        self.max_norm1 = max_norm1
        self.max_norm2 = max_norm2
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, temp_kernel), padding='same', bias=False),
            nn.BatchNorm2d(f1)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (num_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1)),
            nn.Dropout(dropout_rate)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16), padding='same', groups=f2, bias=False),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2)),
            nn.Dropout(dropout_rate)
        )
        
        if self.attention_type == 'agfl':
            self.attn_blocks = nn.ModuleList([AGFL(dim=f2, heads=4, K=3, separate_W=True)])
        elif self.attention_type == 'standard':
            self.attn_blocks = nn.ModuleList([StandardAttention(dim=f2, heads=4)])
        else:
            self.attn_blocks = None

        out_features = self._calculate_out_features(time_points, num_channels)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_features, num_classes)

        self.clip_weights()

    def _calculate_out_features(self, time_points, channels):
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, channels, time_points)
            x = self.block1(dummy_x)
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
            return x.numel()

    def _apply_max_norm(self, layer, max_norm):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def clip_weights(self):
        self._apply_max_norm(self.block2[0], self.max_norm1)
        self._apply_max_norm(self.fc, self.max_norm2)

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
                        
        x = self.flatten(x)
        out = self.fc(x)
        return out