import pytest
from src.localization.iou import box_iou


def test_identical_boxes_have_iou_one():
    box = [10, 10, 50, 50]
    assert box_iou(box, box) == pytest.approx(1.0)


def test_non_overlapping_boxes_have_iou_zero():
    box_a = [0, 0, 10, 10]
    box_b = [20, 20, 30, 30]
    assert box_iou(box_a, box_b) == pytest.approx(0.0)


def test_partially_overlapping_boxes():
    box_a = [0, 0, 10, 10]
    box_b = [5, 5, 15, 15]
    assert box_iou(box_a, box_b) == pytest.approx(25 / 175)


def test_iou_is_between_zero_and_one():
    box_a = [0, 0, 12, 12]
    box_b = [4, 4, 16, 16]
    result = box_iou(box_a, box_b)
    assert 0.0 <= result <= 1.0