import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from agfl_layer import AGFL
from standard_attention import StandardAttention

class CConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(CConv1d, self).__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.pad, **kwargs)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out


class SpikingEEGNet(nn.Module):
    def __init__(self, num_channels=44, num_classes=4, samples=1000, 
                 attention_type='standard', dropout_rate=0.5, beta=0.9):
        super(SpikingEEGNet, self).__init__()
        
        self.attention_type = attention_type
        self.samples = samples
        spike_grad = surrogate.fast_sigmoid(slope=25)
 
        self.conv1 = CConv1d(num_channels, 32, kernel_size=17, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.conv2 = CConv1d(32, 64, kernel_size=17, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.conv3 = CConv1d(64, 64, kernel_size=17, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Attention Setup
        if self.attention_type == 'agfl':
            self.attn_blocks = nn.ModuleList([AGFL(dim=64, heads=4, K=2, separate_W=True)])
        elif self.attention_type == 'standard':
            self.attn_blocks = nn.ModuleList([StandardAttention(dim=64, heads=4)])
        else:
            self.attn_blocks = None
            
        self.lif_attn = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.ffn_linear1 = nn.Linear(64, 128)
        self.lif_ffn1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.ffn_linear2 = nn.Linear(128, 64)
        self.lif_ffn2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc = nn.Linear(64, num_classes)
        self.lif_out = snn.Leaky(beta=1.0, reset_mechanism="none")

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        spk1_record = []
        
        for t in range(syn1.size(2)):
            spk1, mem1 = self.lif1(syn1[:, :, t], mem1)
            spk1_record.append(spk1)
            
        spk1_seq = torch.stack(spk1_record, dim=2)
        
        syn2 = self.pool2(self.bn2(self.conv2(spk1_seq)))
        mem2 = self.lif2.init_leaky()
        spk2_record = []
        
        for t in range(syn2.size(2)):
            spk2, mem2 = self.lif2(syn2[:, :, t], mem2)
            spk2_record.append(spk2)
            
        spk2_seq = torch.stack(spk2_record, dim=2)
        
        syn3 = self.pool3(self.bn3(self.conv3(spk2_seq)))
        mem3 = self.lif3.init_leaky()
        spk3_record = []
        
        for t in range(syn3.size(2)):
            spk3, mem3 = self.lif3(syn3[:, :, t], mem3)
            spk3_record.append(spk3)
            
        spk_seq = torch.stack(spk3_record, dim=1) 
        
        if self.attn_blocks is not None:
            L = len(self.attn_blocks)
            attn_out = spk_seq
            
            for i, block in enumerate(self.attn_blocks):
                if self.attention_type == 'agfl':
                    attn_out = block(attn_out, layer_idx=i, L=L)
                else:
                    attn_out = block(attn_out)
                    
            mem_attn = self.lif_attn.init_leaky()
            mem_ffn1 = self.lif_ffn1.init_leaky()
            mem_ffn2 = self.lif_ffn2.init_leaky()
            smha_record = []
            
            for t in range(spk_seq.size(1)):
                cur_attn = attn_out[:, t, :]
                spk_attn, mem_attn = self.lif_attn(cur_attn, mem_attn)
                
                cur_ffn1 = self.dropout(self.ffn_linear1(spk_attn))
                spk_ffn1, mem_ffn1 = self.lif_ffn1(cur_ffn1, mem_ffn1)
                
                cur_ffn2 = self.dropout(self.ffn_linear2(spk_ffn1))
                spk_ffn2, mem_ffn2 = self.lif_ffn2(cur_ffn2, mem_ffn2)
                
                smha_record.append(spk_ffn2)
                
            spk_seq = torch.stack(smha_record, dim=1)
            

        fc_out = self.fc(spk_seq)
        mem_out = self.lif_out.init_leaky()
        
        for t in range(spk_seq.size(1)):
            _, mem_out = self.lif_out(fc_out[:, t, :], mem_out)
            
        return mem_out