
import numpy as np
import pytest

from src.localization.heatmap import (
    colorize_heatmap,
    overlay_heatmap,
    resize_heatmap,
)


def test_resize_heatmap_returns_requested_dimensions():
    heatmap = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)

    result = resize_heatmap(heatmap, (8, 10))

    assert result.shape == (8, 10)
    assert result.dtype == np.float32
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_resize_heatmap_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="dimensions must be positive"):
        resize_heatmap(np.ones((2, 2)), (0, 4))


def test_colorize_heatmap_returns_rgb_uint8():
    heatmap = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

    result = colorize_heatmap(heatmap)

    assert result.shape == (1, 3, 3)
    assert result.dtype == np.uint8
    assert result[0, 0].tolist() == [0, 0, 0]
    assert result[0, 2].tolist() == [255, 255, 255]


def test_colorize_heatmap_rejects_non_finite_values():
    heatmap = np.array([[0.0, np.nan]])

    with pytest.raises(ValueError, match="finite"):
        colorize_heatmap(heatmap)


def test_overlay_heatmap_returns_rgb_image():
    image = np.zeros((8, 8), dtype=np.uint8)
    heatmap = np.ones((4, 4), dtype=np.float32)

    result = overlay_heatmap(image, heatmap, alpha=0.5)

    assert result.shape == (8, 8, 3)
    assert result.dtype == np.uint8


def test_overlay_heatmap_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        overlay_heatmap(
            np.zeros((4, 4), dtype=np.uint8),
            np.ones((4, 4), dtype=np.float32),
            alpha=1.5,
        )
