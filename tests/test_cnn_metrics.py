import pytest

from scripts.train_cnn import _confusion_metrics


def test_confusion_metrics_calculates_expected_values():
    labels = [0, 0, 0, 1, 1, 1]
    predictions = [0, 0, 1, 1, 1, 0]

    metrics = _confusion_metrics(labels, predictions)

    assert metrics["confusion_matrix"] == [[2, 1], [1, 2]]
    assert metrics["sensitivity"] == pytest.approx(2 / 3)
    assert metrics["specificity"] == pytest.approx(2 / 3)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["f1_score"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx(2 / 3)


def test_confusion_metrics_rejects_empty_inputs():
    with pytest.raises(ValueError, match="empty"):
        _confusion_metrics([], [])


def test_confusion_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        _confusion_metrics([0, 1], [1])