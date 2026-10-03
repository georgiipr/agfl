# pre_norm and dropout reproduce the dataset authors' transformer block
# (LayerNorm before attention, dropout after its output projection). Both are
# off by default so that agfl, mha and hcann are compared at the same position
# of the same backbone with the same surrounding normalization and dropout.
DEFAULTS = {'heads': 4, 'dropout': 0.0, 'pre_norm': False}
