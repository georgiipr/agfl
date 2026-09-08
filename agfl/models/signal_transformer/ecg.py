"""ECG: explicitly combine leads while producing temporal patch tokens."""

import math

from torch import nn
from torch.nn import functional as F

from .backbone import SignalBackbone, validate_common


class TemporalTokenizer(nn.Module):
    def __init__(self, channels, dim, patch_size):
        super().__init__()
        if not isinstance(patch_size, int) or patch_size < 1:
            raise ValueError("ecg_patch_size must be a positive integer")
        self.patch_size = patch_size
        self.projection = nn.Conv1d(channels, dim, patch_size, stride=patch_size)

    def forward(self, x):
        padding = (-x.shape[-1]) % self.patch_size
        if padding:
            # Only the last fixed-size patch is padded; no all-padding tokens.
            x = F.pad(x, (0, padding))
        return self.projection(x).transpose(1, 2)


class ECGSignalTransformer(SignalBackbone):
    token_axis = "time_patch"
    model_variant = "ecg"

    def __init__(self, options, metadata, mixer_factory):
        validate_common(options)
        tokenizer = TemporalTokenizer(int(metadata["channels"]), options["dim"], options["ecg_patch_size"])
        num_tokens = math.ceil(int(metadata["samples"]) / options["ecg_patch_size"])
        super().__init__(options, metadata, mixer_factory, tokenizer, num_tokens)


ECGModel = ECGSignalTransformer
