from .. import AttentionSpec
from .config import DEFAULTS
from .layer import MultiHeadAttention

SPEC = AttentionSpec('mha', DEFAULTS, lambda options, tokens: MultiHeadAttention(
    options['dim'], options['heads'], temporal_bias=options['temporal_bias'],
    output_gate=options['output_gate']))
