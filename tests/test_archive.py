"""Keep independent reference sources traceable to the original commit."""
from pathlib import Path
import subprocess
import pytest

BASELINE = 'fe1621311465e54ccd42d48bdcea90b7fe93cf1b'
ROOT = Path(__file__).resolve().parents[1]
REFERENCES = {
    'agfl.py': 'depricated/agfl_layer_0.py',
    'standard_attention.py': 'standard_attention.py',
    **{f'models/{name}.py': f'models/{name}.py'
       for name in ('eegnet', 'eegencoder', 'dsts_eeg_encoder', 'conformer')},
}


@pytest.mark.parametrize('reference,source', REFERENCES.items())
def test_reference_only_changes_import_paths_and_rmsnorm_compatibility(reference, source):
    try:
        original = subprocess.check_output(['git', 'show', f'{BASELINE}:{source}'], cwd=ROOT, text=True,
                                           stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.fail(f'Original Git commit is required for source parity ({source}): {error}')
    original = original.replace('from depricated.agfl_layer_0 import', 'from tests.references.agfl import')
    original = original.replace('from standard_attention import', 'from tests.references.standard_attention import')
    if source == 'models/dsts_eeg_encoder.py':
        original = original.replace('import torch.nn.functional as F',
                                    'import torch.nn.functional as F\nfrom tests.references.compat import RMSNorm')
        original = original.replace('nn.RMSNorm(', 'RMSNorm(')
    assert (ROOT / 'tests' / 'references' / reference).read_text() == original
