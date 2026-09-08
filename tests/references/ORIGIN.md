# Independent test references

These four original model sources, the active AGFL layer and StandardAttention
come from commit `fe1621311465e54ccd42d48bdcea90b7fe93cf1b`. They are excluded from
the installed package and cannot be selected by the training CLI. They remain
only as independent evidence for migration and numerical checks.

`agfl.py` is the original `depricated/agfl_layer_0.py`, which the original
backbones actually imported. The inactive root `agfl_layer.py` is not the reference.
Model filenames retain their original names. Only their imports were relocated;
DSTS also imports a mathematical RMSNorm compatibility implementation.
`tests/test_archive.py` checks these changes against the Git commit.

The production models are under `agfl/models/`. Their default EEG variants now
mix electrode tokens, requiring changes to feature extraction and readout.
EEGEncoder's repeated TCN residual evaluation is corrected, and DSTS receives
explicit layer indices for the AGFL schedule. No universal original-backbone
parity is claimed. `test_reference.py` checks frozen-weight outputs and input
gradients for EEGNet's explicit temporal variant and Conformer ECG.
`test_models.py` independently checks the AGFL mathematics and gradients.
These checks must be executed on the target machine; they have not run locally.
