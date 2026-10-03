from .. import AttentionSpec
from .config import DEFAULTS
from .layer import HCANNAttention

SPEC = AttentionSpec('hcann', DEFAULTS, lambda options, tokens: HCANNAttention(
    options['dim'], options['heads'], dropout=options['dropout'], pre_norm=options['pre_norm']))
