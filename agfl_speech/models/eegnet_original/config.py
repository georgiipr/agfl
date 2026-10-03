# The dataset authors' EEGNet (their MyEEGNet/models/EEGNetModel.py): a 32-sample
# temporal kernel, F1=16, D=2, F2=32, pooling 8 then 16, dropout 0.5, and the
# max-norm constraint (1.0 on the spatial filter, 0.25 on the classifier)
# applied once when the model is built. `max_norm_schedule: every_step` instead
# re-applies the constraint after every optimizer step, as the project EEGNet does.
DEFAULTS = {
    'temp_kernel': 32, 'f1': 16, 'd': 2, 'f2': 32, 'pk1': 8, 'pk2': 16,
    'dropout_rate': .5, 'max_norm1': 1., 'max_norm2': .25, 'max_norm_schedule': 'init',
    # Fixed: the attention sits before the authors' spatial filter. Declaring it
    # lets the configuration check that the heads divide F1 before any launch.
    'electrode_architecture': 'pre_spatial',
    'attention_residual': True, 'attention_dropout': 0.0,
    'batch_norm_momentum': .1, 'batch_norm_eps': 1e-5,
}
