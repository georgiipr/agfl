import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from agfl_layer import AGFL
from standard_attention import StandardAttention

class SpikingEEGNet(nn.Module):
    def __init__(self, num_channels=44, num_classes=4, samples=1000, 
                 attention_type='standard', dropout_rate=0.5, beta=0.9):
        super(SpikingEEGNet, self).__init__()
        
        self.attention_type = attention_type
        self.samples = samples
        
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=17, padding=8, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(32, 64, kernel_size=17, padding=8, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(64, 64, kernel_size=17, padding=8, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.lif_input = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        if self.attention_type == 'agfl':
            self.attn_blocks = nn.ModuleList([AGFL(dim=64, heads=4, K=2, separate_W=True)])
        elif self.attention_type == 'standard':
            self.attn_blocks = nn.ModuleList([StandardAttention(dim=64, heads=4)])
        else:
            self.attn_blocks = None
            
        self.lif_attn = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.ffn = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.lif_ffn = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        
        batch_size, channels, time_steps = x.size()
        
        mem_input = self.lif_input.init_leaky()
        spk_record = []
        
        for t in range(time_steps):
            cur = x[:, :, t]
            spk, mem_input = self.lif_input(cur, mem_input)
            spk_record.append(spk)
            
        spk_seq = torch.stack(spk_record, dim=1)
        
        if self.attn_blocks is not None:
            L = len(self.attn_blocks)
            attn_out = spk_seq
            
            for i, block in enumerate(self.attn_blocks):
                if self.attention_type == 'agfl':
                    attn_out = block(attn_out, layer_idx=i, L=L)
                else:
                    attn_out = block(attn_out)
                    
            mem_attn = self.lif_attn.init_leaky()
            mem_ffn = self.lif_ffn.init_leaky()
            smha_record = []
            
            for t in range(time_steps):
                cur_attn = attn_out[:, t, :]
                spk_attn, mem_attn = self.lif_attn(cur_attn, mem_attn)
                
                cur_ffn = self.ffn(spk_attn)
                spk_ffn, mem_ffn = self.lif_ffn(cur_ffn, mem_ffn)
                
                spk_ffn = self.dropout(spk_ffn)
                smha_record.append(spk_ffn)
                
            spk_seq = torch.stack(smha_record, dim=1)
            
        pooled = spk_seq.mean(dim=1) 
        
        out = self.fc(pooled)
        return out