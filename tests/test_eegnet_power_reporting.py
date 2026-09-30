"""Target-machine checks for class errors and the fixed EEGNet power study."""
from copy import deepcopy

import numpy as np
import pytest

from agfl.analysis import class_statistics
from agfl.config import comparison_identity, experiment_identity, merge, resolve_experiments
from agfl.metrics import classification_metrics
from tests.references.presets import load_archived_preset as load_preset, archived_experiments
from agfl.visualization.temporal_statistics import TemporalStatisticsCapture, temporal_statistics_figures


def test_per_class_metrics_expose_confusion_and_undefined_precision():
    result = classification_metrics([0, 0, 1, 2],
                                    [[.9, .05, .05], [.1, .8, .1], [.1, .7, .2], [.1, .8, .1]])
    assert result['accuracy'] == .5
    assert result['confusion_matrix'] == [[1, 1, 0], [0, 1, 0], [0, 1, 0]]
    assert result['per_class'][0]['recall'] == .5
    assert result['per_class'][1]['precision'] == pytest.approx(1 / 3)
    assert result['per_class'][2]['recall'] == 0
    assert result['per_class'][2]['precision'] is None
    assert result['f1'] == pytest.approx(np.mean([row['f1'] for row in result['per_class']]))
    absent = classification_metrics([0, 0], [[.9, .1], [.8, .2]])
    assert absent['per_class'][1]['recall'] is None
    with pytest.raises(ValueError, match='Expected nonempty labels'):
        classification_metrics([[0], [1]], [[.9, .1], [.8, .2]])


def test_class_summaries_average_seed_rates_not_repeated_trial_counts():
    first = classification_metrics([0] * 9, [[.9, .1]] * 9)
    second = classification_metrics([0], [[.1, .9]])
    runs = []
    for seed, metrics in enumerate((first, second)):
        runs.append({'experiment_id': 'e', 'comparison_id': 'p', 'dataset': 'eeg', 'seed': seed,
                     'split_id': str(seed), 'backbone_key': 'eegnet', 'attention_key': 'agfl',
                     'config': {'subject_id': 'A03', 'resolved_metadata': {'label_names': ['feet', 'tongue']}},
                     'validation': metrics, 'test': metrics})
    rows, summaries = class_statistics(runs)
    assert len(rows) == 8
    feet = next(row for row in summaries if row['partition'] == 'test' and row['class_id'] == 0)
    assert feet['recall_mean'] == .5  # Not nine correct repeated occurrences / ten.
    assert feet['recall_std'] == pytest.approx(np.sqrt(.5))
    assert feet['support_occurrences'] == 10
    assert feet['n_seeds'] == feet['recall_n'] == 2
    absent = next(row for row in summaries if row['partition'] == 'test' and row['class_id'] == 1)
    assert absent['recall_mean'] is None and absent['recall_n'] == 0
    for run in runs:
        run['validation'] = run['test'] = {'accuracy': .5}
    assert class_statistics(runs) == ([], [])


def test_power_preset_keeps_data_and_agfl_recipe_and_matches_all_attentions():
    previous = load_preset('eegnet-a03-conditioned-agfl')
    current = load_preset('eegnet-a03-power-ema')
    assert current['base']['data'] == previous['base']['data']
    assert current['base']['split'] == previous['base']['split']
    assert current['experiments'][0] == previous['experiments'][0]
    assert current['experiments'][2] == previous['experiments'][2]
    assert [row['attention'] for row in current['experiments']] == ['agfl', 'mha', 'linformer']
    configs = [archived_experiments(merge(current['base'], experiment))[0] for experiment in current['experiments']]
    assert len({comparison_identity(config) for config in configs}) == 1
    assert len({experiment_identity(config) for config in configs}) == 3
    assert all(config['subject_id'] == 'A03' and config['model'] == 'eegnet' for config in configs)
    assert sum(len(config['seeds']) for config in configs) == 15
    for config in configs:
        options, training = config['model_options'], config['training']
        assert options['temp_kernel'] == 125 and options['temporal_statistics'] == 'mean_logvar'
        assert 1000 // options['pk1'] // options['pk2'] == 7
        assert training['epochs'] == 250 and training['ema_decay'] == .99
        assert training['loss'] == 'cross_entropy' and training['checkpoint_tiebreaker'] == 'f1_loss'
        assert training['augmentation']['shift'] == 25 and training['augmentation']['scale'] == .1


def test_temporal_diagnostics_keep_all_batches_and_trial_axes(tmp_path):
    import torch
    from agfl.models.eegnet.reproduction import TemporalStatisticsFusion
    from tests.test_agfl_conditioning_diagnostics import DiagnosticWriter
    module = TemporalStatisticsFusion(features=4, out_features=4, bins=3, samples=20)
    with torch.no_grad():
        module.projection.weight.copy_(torch.eye(4))
    model = torch.nn.Module()
    model.temporal_statistics = module
    generator = torch.Generator().manual_seed(141)
    inputs = torch.randn(10, 4, 1, 20, generator=generator)
    inputs *= torch.arange(1, 41, dtype=torch.float32).reshape(10, 4, 1, 1) / 10
    ids = [f'trial_{i}' for i in range(5)]
    before = deepcopy(module.state_dict())
    with torch.no_grad():
        expected_logvar = module.log_variance(inputs).reshape(5, 2, 4, 3).numpy()
        expected_addition = module(inputs).reshape(5, 2, 4, 3).numpy()
    capture = TemporalStatisticsCapture(model)
    try:
        for start, stop in [(0, 2), (2, 5)]:
            capture.start_batch(ids[start:stop])
            with torch.no_grad():
                module(inputs[2 * start:2 * stop])
    finally:
        for handle in capture.hooks:
            handle.remove()
    summary = temporal_statistics_figures(DiagnosticWriter(tmp_path), capture, ids, [0, 1, 0, 1, 0],
                                         ['feet', 'tongue'], 'validation')
    assert summary['samples'] == 5
    with np.load(tmp_path / summary['array_file'], allow_pickle=False) as arrays:
        assert arrays['log_variance'].shape == (5, 2, 4, 3)
        assert arrays['additive_features'].shape == (5, 2, 4, 3)
        assert arrays['sample_ids'].tolist() == ids
        assert arrays['bin_edges'].tolist() == list(module.bin_edges)
        np.testing.assert_allclose(arrays['log_variance'], expected_logvar, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(arrays['additive_features'], expected_addition, rtol=1e-6, atol=1e-6)
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    capture.ids[0] = capture.ids[0][::-1]
    with pytest.raises(ValueError, match='IDs/order'):
        temporal_statistics_figures(DiagnosticWriter(tmp_path), capture, ids, [0] * 5, ['feet'], 'validation')
