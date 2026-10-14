
import numpy as np
import pytest

from src.localization.bounding_box import (
    activated_mask,
    clusters_to_boxes,
    filter_clusters_two_sigma,
)
from src.localization.iou import box_iou, image_iou


def test_identical_boxes_have_iou_one():
    box = (10.0, 20.0, 30.0, 40.0)
    assert box_iou(box, box) == pytest.approx(1.0)


def test_non_overlapping_boxes_have_iou_zero():
    assert box_iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_partially_overlapping_boxes():
    first = (0, 0, 10, 10)
    second = (5, 0, 10, 10)
    assert box_iou(first, second) == pytest.approx(1 / 3)


def test_image_iou_handles_empty_inputs():
    assert image_iou([], []) == 0.0


def test_image_iou_penalizes_missing_predictions():
    truth = [(0, 0, 10, 10)]
    assert image_iou([], truth) == 0.0


def test_clusters_to_boxes_scales_coordinates():
    clusters = [[(1, 2), (2, 3)]]
    boxes = clusters_to_boxes(
        clusters,
        map_size=(4, 4),
        image_size=(8, 8),
        apply_two_sigma=False,
    )
    assert boxes == [(4.0, 2.0, 4.0, 4.0)]


def test_clusters_to_boxes_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="dimensions must be positive"):
        clusters_to_boxes(
            [[(0, 0)]],
            map_size=(0, 4),
            image_size=(8, 8),
        )


def test_activated_mask_uses_inclusive_threshold():
    cam = np.array([[0.2, 0.5], [0.7, 1.0]])
    result = activated_mask(cam, 0.5)
    assert result.tolist() == [
        [False, True],
        [True, True],
    ]


def test_activated_mask_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="threshold"):
        activated_mask(np.ones((2, 2)), 0.0)


def test_filter_clusters_returns_empty_for_empty_input():
    assert filter_clusters_two_sigma([]) == []
