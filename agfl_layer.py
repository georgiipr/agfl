import math
import torch
import torch.nn as nn

class GraphConstructor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.scale = math.sqrt(dim) 

    def forward(self, X):
        tau = self.temperature.clamp(0.1, 5.0)
        scores = (X @ X.transpose(-1, -2)) / (self.scale * tau)
        return scores

class GraphFilter(nn.Module):
    def __init__(self, dim, K, separate_W=True):
        super().__init__()
        self.K = K
        self.separate_W = separate_W
        
        self.alpha_logits = nn.Parameter(torch.randn(K + 1) * 0.02)
        
        if separate_W:
            self.W = nn.ModuleList([
                nn.Linear(dim, dim, bias=False)
                for _ in range(K + 1)
            ])
        else:
            self.W = nn.Linear(dim, dim, bias=False)

    def forward(self, A, X):
        alpha = torch.softmax(self.alpha_logits, dim=0)
        
        P_k = X 
        if self.separate_W:
            H = alpha[0] * self.W[0](P_k)
        else:
            H = alpha[0] * self.W(P_k)
            
        for k in range(1, self.K + 1):
            P_k = A @ P_k
            
            if self.separate_W:
                proj = self.W[k](P_k)
            else:
                proj = self.W(P_k)
                
            H = H + alpha[k] * proj
            
        return H

class AGFL(nn.Module):
    def __init__(self, dim, heads, K, separate_W=True, causal=True):
        super().__init__()
        self.heads = heads
        self.dim_h = dim // heads
        self.causal = causal
        
        self.builders = nn.ModuleList([
            GraphConstructor(self.dim_h)
            for _ in range(heads)
        ])
        self.filters = nn.ModuleList([
            GraphFilter(self.dim_h, K, separate_W)
            for _ in range(heads)
        ])
        
        self.ln = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.last_adj = None

    def forward(self, X, **kwargs):
        B, N, D = X.shape

        X = X.view(B, N, self.heads, self.dim_h).transpose(1, 2)
        
        outs = []
        adjs = []
        
        for h in range(self.heads):
            Xh = X[:, h]
            
            S = self.builders[h](Xh)
            
            if self.causal:
                mask = torch.triu(torch.full((N, N), float('-inf'), device=X.device), diagonal=1)
                S = S + mask
            
            A_dense = torch.softmax(S, dim=-1)
            
            if self.causal:
                A_dense = torch.nan_to_num(A_dense, nan=0.0)
            
            outs.append(self.filters[h](A_dense, Xh))
            adjs.append(A_dense.detach())

        self.last_adj = torch.stack(adjs)
        
        out = torch.cat(outs, dim=-1)

        return self.proj(self.ln(out))