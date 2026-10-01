TRAINING_DEFAULTS = {
    # EEG follows the reference EEGNet recipe: cross-entropy with an Adam-family
    # optimizer at 1e-3. The budget is longer because each subject trains on
    # roughly 170 trials (about three minibatches per epoch), and the checkpoint
    # is the minimum validation loss rather than the maximum of a 55-trial
    # accuracy over hundreds of epochs.
    'eeg': {'epochs': 500, 'learning_rate': 1e-3, 'loss': 'cross_entropy',
            'checkpoint_criterion': 'loss'},
    'ecg': {'epochs': 50, 'learning_rate': 3e-4, 'focal_gamma': 1.0},
}
