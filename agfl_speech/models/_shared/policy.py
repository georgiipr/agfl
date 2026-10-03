TRAINING_DEFAULTS = {
    # EEG follows the reference EEGNet recipe: cross-entropy with an Adam-family
    # optimizer at 1e-3, and the checkpoint is the minimum validation loss
    # rather than the maximum of a small-sample accuracy. The 500-epoch default
    # is the AGFL project's value; a pooled SI_Hom cohort trains on about 1580
    # trials (about 25 minibatches of 64 per epoch) and an individual subject on
    # 130 to 300 trials, so every preset states its own epoch budget.
    'eeg': {'epochs': 500, 'learning_rate': 1e-3, 'loss': 'cross_entropy',
            'checkpoint_criterion': 'loss'},
}
