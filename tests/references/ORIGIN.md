# Independent test reference

`agfl.py` is the original AGFL layer of the AGFL project (its
`depricated/agfl_layer_0.py`, commit `fe1621311465e54ccd42d48bdcea90b7fe93cf1b`
of that repository). It is not part of the `agfl_speech` package and cannot be
selected by the training CLI. `test_models.py` uses it as independent evidence
that the packaged AGFL attention keeps the original mathematics and gradients.
These checks must be executed on the target machine; they have not run locally.
