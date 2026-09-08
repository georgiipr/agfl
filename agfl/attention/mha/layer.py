from ..projections import ProjectedMixer

class MultiHeadAttention(ProjectedMixer):
    def forward(self, x):
        q, k, v = self.project(x)
        weights = (q @ k.transpose(-1, -2) / self.head_dim**.5).softmax(-1)
        self.last_attn = weights.detach()
        return self.combine(weights @ v)
