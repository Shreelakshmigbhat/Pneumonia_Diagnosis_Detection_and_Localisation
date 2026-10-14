import pytest
from scripts.train_cnn import _confusion_metrics


def test_perfect_predictions_have_perfect_metrics():
    labels = [0, 0, 1, 1]
    predictions = [0, 0, 1, 1]

    result = _confusion_metrics(labels, predictions)

    assert result["sensitivity"] == pytest.approx(1.0)
    assert result["specificity"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["f1_score"] == pytest.approx(1.0)
    assert result["balanced_accuracy"] == pytest.approx(1.0)


def test_metrics_for_mixed_predictions():
    labels = [1, 1, 0, 0]
    predictions = [1, 0, 1, 0]

    result = _confusion_metrics(labels, predictions)

    assert result["sensitivity"] == pytest.approx(0.5)
    assert result["specificity"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(0.5)
    assert result["f1_score"] == pytest.approx(0.5)
    assert result["balanced_accuracy"] == pytest.approx(0.5)


def test_mismatched_lengths_raise_error():
    with pytest.raises(ValueError):
        _confusion_metrics([0, 1], [0])


def test_empty_labels_raise_error():
    with pytest.raises(ValueError):
        _confusion_metrics([], [])