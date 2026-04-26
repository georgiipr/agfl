import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from depricated.agfl_layer_0 import AGFL
from standard_attention import StandardAttention

class SpikingEEGNet(nn.Module):
    def __init__(self, num_channels=44, num_classes=4, samples=1000, 
                 f1=16, f2=32, attention_type='standard', 
                 dropout_rate=0.5, beta=0.9):
        super(SpikingEEGNet, self).__init__()
        
        self.attention_type = attention_type
        self.samples = samples
        
        # surrogate gradient allows backpropagation through non-differentiable spikes
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.conv1 = nn.Conv1d(1, f1, kernel_size=num_channels, bias=False)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.conv2 = nn.Conv1d(f1, f2, kernel_size=1, bias=False)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        #attention options - refer to agfl_layer.py and standard_attention.py
        if self.attention_type == 'agfl':
            self.attn_blocks = nn.ModuleList([AGFL(dim=f2, heads=4, K=2, separate_W=True)])
        elif self.attention_type == 'standard':
            self.attn_blocks = nn.ModuleList([StandardAttention(dim=f2, heads=4)])
        else:
            self.attn_blocks = None
            
        self.fc = nn.Linear(f2, num_classes)

    def forward(self, x):
        # expect shapes: (Batch, Channels=44 - since 22 x 2 channels, Time=1000)
        batch_size, _, time_steps = x.size()
        
        #  membrabe potentials for the sequence
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk_record = []
        
        for t in range(time_steps):
            x_t = x[:, :, t].unsqueeze(1)
            
            cur1 = self.conv1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2 = self.dropout(spk2)
            spk_record.append(spk2)
            
        spk_seq = torch.stack(spk_record, dim=-1).squeeze(2)
        
        spk_seq = spk_seq.transpose(1, 2)
        
        if self.attn_blocks is not None:
            L = len(self.attn_blocks)
            for i, block in enumerate(self.attn_blocks):
                if self.attention_type == 'agfl':
                    spk_seq = block(spk_seq, layer_idx=i, L=L)
                else:
                    spk_seq = block(spk_seq)
                    
        # receive a single vector per batch via polling
        pooled = spk_seq.mean(dim=1) 
        
        out = self.fc(pooled)
        return out