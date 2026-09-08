from .. import AttentionSpec
from .config import DEFAULTS
from .layer import Nystromformer

SPEC = AttentionSpec('nystromformer', DEFAULTS, lambda options, tokens: Nystromformer(options))
