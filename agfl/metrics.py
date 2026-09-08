"""Metrics from held-out probabilities with explicit undefined-AUC reporting."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def classification_metrics(targets, probabilities):
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or len(targets) != len(probabilities) or not len(targets):
        raise ValueError("Expected nonempty labels and [samples, classes] probabilities")
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities are nonfinite")
    if probabilities.shape[1] < 2 or targets.min() < 0 or targets.max() >= probabilities.shape[1]:
        raise ValueError('Targets must be valid class indices for at least two classes')
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(-1), 1, atol=1e-5):
        raise ValueError('Metrics require normalized class probabilities')
    classes = np.arange(probabilities.shape[1])
    result = {
        "accuracy": float(accuracy_score(targets, probabilities.argmax(axis=1))),
        "f1": float(f1_score(targets, probabilities.argmax(axis=1), labels=classes, average="macro", zero_division=0)),
        "roc_auc": None, "roc_auc_reason": None, "n_samples": len(targets),
        "class_counts": np.bincount(targets, minlength=len(classes)).tolist(),
    }
    if len(np.unique(targets)) != len(classes):
        result["roc_auc_reason"] = "Held-out partition does not contain every class"
    elif len(classes) == 2:
        result["roc_auc"] = float(roc_auc_score(targets, probabilities[:, 1]))
    else:
        result["roc_auc"] = float(roc_auc_score(targets, probabilities, labels=classes, multi_class="ovr", average="macro"))
    return result
