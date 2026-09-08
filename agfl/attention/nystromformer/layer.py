"""Segment landmark Nyström attention, Xiong et al., arXiv:2102.03902.

Uses the Moore-Penrose inverse directly for the small landmark matrix. A fixed
number of landmarks yields linear sequence scaling; this is not a full BERT
reproduction and adds no baseline-only convolutional residual.
"""
import torch
from ..projections import ProjectedMixer

class Nystromformer(ProjectedMixer):
    def __init__(self, options):
        super().__init__(options['dim'], options['heads'])
        self.landmarks = options['landmarks']
        self.pinv_rtol = options.get('pinv_rtol', 1e-5)
        if type(self.landmarks) is not int or self.landmarks < 1:
            raise ValueError('landmarks must be positive')
        if not 0 < self.pinv_rtol < 1:
            raise ValueError('pinv_rtol must be between zero and one')

    def forward(self, x):
        q, k, v = self.project(x)
        original_dtype = v.dtype
        compute_dtype = torch.float64 if v.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            mixed = self.mix(q.to(compute_dtype), k.to(compute_dtype), v.to(compute_dtype))
        return self.combine(mixed.to(original_dtype))

    def mix(self, q, k, v):
        q, k = q * self.head_dim**-.25, k * self.head_dim**-.25
        n = q.shape[-2]
        m = min(n, self.landmarks)
        if m == n:
            # Exact full-landmark limit, avoiding avoidable inverse roundoff.
            return (q @ k.transpose(-1, -2)).softmax(-1) @ v
        q_land = torch.stack([part.mean(-2) for part in torch.tensor_split(q, m, dim=-2)], dim=-2)
        k_land = torch.stack([part.mean(-2) for part in torch.tensor_split(k, m, dim=-2)], dim=-2)
        left = (q @ k_land.transpose(-1, -2)).softmax(-1)
        middle = (q_land @ k_land.transpose(-1, -2)).softmax(-1)
        right = (q_land @ k.transpose(-1, -2)).softmax(-1)
        inverse = torch.linalg.pinv(middle, rtol=self.pinv_rtol)
        return left @ (inverse @ (right @ v))
