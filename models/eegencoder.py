import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from agfl_layer import AGFL
from standard_none_attention import StandardAttention, NoAttention

class Conv1dL2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)
    def forward(self, x): return self.conv1(x)

class Conv2dL2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)
    def forward(self, x): return self.conv2(x)

class ConvBlock(nn.Module):
    def __init__(self, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22, dropout=0.5):
        super().__init__()
        F2 = F1 * D
        self.conv1 = Conv2dL2(1, F1, (kernLength, 1), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.depthwise = Conv2dL2(F1, F2, (1, in_chans), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F2)
        self.activation = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((8, 1))
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = Conv2dL2(F2, F2, (16, 1), padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((poolSize, 1))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1).permute(0, 1, 3, 2) # ensure [B, 1, T, C]
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        return x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock_(nn.Module):
    def __init__(self, input_dimension, depth, kernel_size, filters, dropout, max_norm=0.6, activation='relu'):
        super().__init__()
        self.activation = getattr(F, activation)
        self.blocks = nn.ModuleList()
        self.downsample = nn.Conv1d(input_dimension, filters, 1) if input_dimension != filters else None
        self.cn1 = nn.Sequential(Conv1dL2(input_dimension, filters, kernel_size), nn.BatchNorm1d(filters), nn.SiLU(), nn.Dropout(0.5))
        self.cn2 = nn.Sequential(Conv1dL2(filters, filters, kernel_size), nn.BatchNorm1d(filters), nn.SiLU(), nn.Dropout(0.5))

        for i in range(depth-1):
            dilation_size = 2 ** (i+1)
            padding = (kernel_size - 1) * dilation_size
            self.blocks.append(nn.Sequential(
                Conv1dL2(filters if i > 0 else input_dimension, filters, kernel_size, padding=padding, dilation=dilation_size),
                Chomp1d(padding), nn.BatchNorm1d(filters), nn.ReLU(), nn.Dropout(dropout),
                Conv1dL2(filters, filters, kernel_size, padding=padding, dilation=dilation_size),
                Chomp1d(padding), nn.BatchNorm1d(filters), nn.ReLU(), nn.Dropout(dropout)
            ))

    def forward(self, x):
        out = x.transpose(1, 2)
        out = self.cn2(self.cn1(out))
        res = self.downsample(out) if self.downsample is not None else out

        for i, block in enumerate(self.blocks):
            out = block(out) + (res if i == 0 else self.blocks[i-1](res))
            out = self.activation(out)
        return out.transpose(1, 2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class EEGEncoder(nn.Module):
    def __init__(self, attention_type='agfl', n_classes=4, in_chans=22, 
                 eegn_F1=16, eegn_D=2, eegn_kernelSize=64, tcn_depth=3, 
                 tcn_kernelSize=4, tcn_filters=32):
        super().__init__()
        self.attention_type = attention_type.lower()
        F2 = eegn_F1 * eegn_D

        self.conv_block = ConvBlock(F1=eegn_F1, kernLength=eegn_kernelSize, poolSize=7, D=eegn_D, in_chans=in_chans, dropout=0.3)
        
        self.tcn_block = TCNBlock_(F2, tcn_depth, tcn_kernelSize, tcn_filters, 0.3, activation='elu')
        self.pe = PositionalEncoding(F2, max_len=50)
        
        if self.attention_type == 'agfl':
            self.attn_blocks = nn.ModuleList([AGFL(dim=F2, heads=4, K=3, separate_W=True)])
        elif self.attention_type == 'standard':
            self.attn_blocks = nn.ModuleList([StandardAttention(dim=F2, heads=4)])
        else:
            self.attn_blocks = None

        #self.attn_blocks = None

        self.aa_drop = nn.Dropout(0.3)
        
        self.classifier = nn.Sequential(
            nn.Linear(tcn_filters, tcn_filters // 2),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(tcn_filters // 2, n_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x[:, :, :, 0].permute(0, 2, 1) 
        
        x_base = self.aa_drop(x)

        tcn_out = self.tcn_block(x_base)[:, -1, :]

        if self.attn_blocks is not None:
            x_attn = self.pe(x_base) 
            
            if self.attention_type == 'agfl':
                attn_out = self.attn_blocks[0](x_attn, layer_idx=0, L=1)
            else:
                attn_out = self.attn_blocks[0](x_attn)
            
            attn_pooled = attn_out.mean(dim=1)
            
            tcn_out = tcn_out + F.dropout(attn_pooled, p=0.3, training=self.training)

        return self.classifier(tcn_out)