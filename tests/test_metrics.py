"""Target-machine checks for per-class metrics and their seed summaries."""
import numpy as np
import pytest

from agfl_speech.analysis import class_statistics
from agfl_speech.metrics import classification_metrics


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
    # A held-out partition without every class has no defined AUC.
    assert absent['roc_auc'] is None and absent['roc_auc_reason']
    with pytest.raises(ValueError, match='Expected nonempty labels'):
        classification_metrics([[0], [1]], [[.9, .1], [.8, .2]])


def test_class_summaries_average_seed_rates_not_repeated_trial_counts():
    first = classification_metrics([0] * 9, [[.9, .1]] * 9)
    second = classification_metrics([0], [[.1, .9]])
    runs = []
    for seed, metrics in enumerate((first, second)):
        runs.append({'experiment_id': 'e', 'comparison_id': 'p', 'dataset': 'si_hom', 'seed': seed,
                     'split_id': str(seed), 'backbone_key': 'eegnet', 'attention_key': 'agfl',
                     'config': {'subject_id': 'S03', 'resolved_metadata': {'label_names': ['P1a', 'P1b']}},
                     'validation': metrics, 'test': metrics})
    rows, summaries = class_statistics(runs)
    assert len(rows) == 8
    first_class = next(row for row in summaries if row['partition'] == 'test' and row['class_id'] == 0)
    assert first_class['class_name'] == 'P1a' and first_class['subject_id'] == 'S03'
    assert first_class['recall_mean'] == .5  # Not nine correct repeated occurrences / ten.
    assert first_class['recall_std'] == pytest.approx(np.sqrt(.5))
    assert first_class['support_occurrences'] == 10
    assert first_class['n_seeds'] == first_class['recall_n'] == 2
    absent = next(row for row in summaries if row['partition'] == 'test' and row['class_id'] == 1)
    assert absent['recall_mean'] is None and absent['recall_n'] == 0
    for run in runs:
        run['validation'] = run['test'] = {'accuracy': .5}
    assert class_statistics(runs) == ([], [])
