from .. import AttentionSpec
from .config import DEFAULTS
from .layer import AGFL

SPEC = AttentionSpec('agfl', DEFAULTS, lambda options, tokens: AGFL(options))
