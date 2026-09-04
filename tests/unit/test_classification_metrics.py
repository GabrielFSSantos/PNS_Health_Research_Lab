import numpy as np

from pns_health_research.modules.classification.metrics import compute_classification_metrics


def test_classification_metrics_positive_class():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    m = compute_classification_metrics(y_true, y_pred)
    assert "f1_positive" in m
    assert "balanced_accuracy" in m
    assert m["f1"] == m["f1_macro"]
