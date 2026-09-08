from .. import AttentionSpec
from .config import DEFAULTS
from .layer import Linformer

SPEC = AttentionSpec('linformer', DEFAULTS, lambda options, tokens: Linformer(options, tokens))
