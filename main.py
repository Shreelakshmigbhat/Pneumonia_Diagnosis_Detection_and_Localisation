"""Run a fast Phase 1 / Phase 2 pneumonia-localization demo."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import (  # noqa: E402
    DEFAULT_STATS_PATH,
    load_normalization_stats,
    preprocess_image,
    read_dicom_pixels,
)
from src.localization.bounding_box import (  # noqa: E402
    activated_mask,
    clusters_to_boxes,
)
from src.localization.cam import compute_cam  # noqa: E402
from src.localization.clustering import find_clusters  # noqa: E402
from src.localization.heatmap import overlay_heatmap  # noqa: E402
from src.localization.improved_cam import compute_gradcam_plus_plus  # noqa: E402
from src.localization.iou import image_iou  # noqa: E402
from src.localization.pipeline import (  # noqa: E402
    LocalizationConfig,
    _draw_box_set,
    _read_source_ground_truth,
    _read_test_positive_images,
    _safe_image_path,
    load_cnn_checkpoint,
)
from src.models.cnn import INPUT_SIZE  # noqa: E402


def _make_boxes(cam: np.ndarray, image_size: tuple[int, int], threshold: float) -> list[tuple[float, float, float, float]]:
    clusters = find_clusters(activated_mask(cam, threshold))
    return clusters_to_boxes(
        clusters,
        map_size=cam.shape,
        image_size=image_size,
        apply_two_sigma=True,
    )


def _render_overlay(
    original: Image.Image,
    cam: np.ndarray,
    ground_truth: list[tuple[float, float, float, float]],
    predictions: list[tuple[float, float, float, float]],
) -> Image.Image:
    overlay_array = overlay_heatmap(
        np.asarray(original, dtype=np.uint8),
        cam,
        alpha=0.45,
    )
    overlay = Image.fromarray(overlay_array, mode="RGB")
    image_size = (original.height, original.width)
    _draw_box_set(
        overlay,
        ground_truth,
        color=(0, 200, 83),
        image_size=image_size,
    )
    _draw_box_set(
        overlay,
        predictions,
        color=(255, 109, 0),
        image_size=image_size,
    )
    return overlay


def run_demo(
    phase: str = "both",
    image_id: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Run selected frozen-checkpoint inference for one positive test image."""
    config = LocalizationConfig(threshold=0.5)
    model, device, checkpoint_info = load_cnn_checkpoint(ROOT)
    if checkpoint_info["checkpoint_epoch"] != 9:
        raise ValueError(
            f"Expected the saved epoch-9 checkpoint; found epoch "
            f"{checkpoint_info['checkpoint_epoch']}"
        )
    stats = load_normalization_stats(ROOT / DEFAULT_STATS_PATH)
    if stats.source_split != "train" or stats.image_size != INPUT_SIZE:
        raise ValueError("Expected saved train-fitted 128x128 preprocessing statistics")

    test_images = _read_test_positive_images(ROOT)
    test_images.sort(key=lambda record: record["image_id"])
    if image_id is None:
        record = test_images[0]
    else:
        record = next(
            (item for item in test_images if item["image_id"] == image_id),
            None,
        )
        if record is None:
            raise ValueError(
                f"{image_id!r} is not a Pneumonia-positive image in the fixed test split"
            )

    image_path = _safe_image_path(ROOT, record["image_path"])
    pixels, pixel_scale = read_dicom_pixels(image_path)
    original = Image.fromarray(
        np.uint8(np.clip(pixels / pixel_scale, 0.0, 1.0) * 255),
        mode="L",
    ).convert("RGB")
    image_tensor = torch.from_numpy(preprocess_image(image_path, stats))
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class = int(logits.argmax(dim=1).item())

    ground_truth = _read_source_ground_truth(ROOT, {record["image_id"]})[
        record["image_id"]
    ]
    destination = output_dir
    if destination is None:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = ROOT / "demo_outputs" / f"{record['image_id']}_{run_stamp}"
    else:
        destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)

    original.save(destination / "input.png")
    summary: dict[str, Any] = {
        "image_id": record["image_id"],
        "true_class": "Pneumonia",
        "predicted_class": "Pneumonia" if predicted_class == 1 else "Healthy",
        "pneumonia_softmax_confidence": float(probabilities[1].item()),
        "checkpoint_epoch": int(checkpoint_info["checkpoint_epoch"]),
        "device": str(device),
        "threshold": config.threshold,
        "two_sigma_filter": config.apply_two_sigma_filter,
        "output_files": ["input.png"],
        "phases": {},
        "note": "Inference demo only; not a clinical diagnosis.",
    }
    image_size = (original.height, original.width)

    if phase in {"phase1", "both"}:
        with torch.inference_mode():
            phase1_map = compute_cam(
                model,
                image_tensor,
                class_index=1,
                output_size=INPUT_SIZE,
            )[0].cpu().numpy()
        phase1_boxes = _make_boxes(phase1_map, image_size, config.threshold)
        phase1_iou = image_iou(phase1_boxes, ground_truth)
        filename = "phase1_cam.png"
        _render_overlay(original, phase1_map, ground_truth, phase1_boxes).save(
            destination / filename
        )
        summary["output_files"].append(filename)
        summary["phases"]["phase1"] = {
            "method": "CNN class activation map",
            "iou": phase1_iou,
            "ground_truth_box_count": len(ground_truth),
            "predicted_box_count": len(phase1_boxes),
        }

    if phase in {"phase2", "both"}:
        phase2_map = compute_gradcam_plus_plus(
            model,
            image_tensor,
            class_index=1,
            output_size=INPUT_SIZE,
            epsilon=1e-8,
        )[0].cpu().numpy()
        phase2_boxes = _make_boxes(phase2_map, image_size, config.threshold)
        phase2_iou = image_iou(phase2_boxes, ground_truth)
        filename = "phase2_gradcam_plus_plus.png"
        _render_overlay(original, phase2_map, ground_truth, phase2_boxes).save(
            destination / filename
        )
        summary["output_files"].append(filename)
        summary["phases"]["phase2"] = {
            "method": "Grad-CAM++",
            "iou": phase2_iou,
            "ground_truth_box_count": len(ground_truth),
            "predicted_box_count": len(phase2_boxes),
        }

    summary_path = destination / "summary.json"
    summary["output_files"].append(summary_path.name)
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Image: {record['image_id']} (fixed Pneumonia-positive test split)")
    print(f"Prediction: {summary['predicted_class']}")
    print(f"Pneumonia confidence: {summary['pneumonia_softmax_confidence']:.4f}")
    for phase_name, result in summary["phases"].items():
        print(
            f"{phase_name}: {result['method']}; IoU={result['iou']:.4f}; "
            f"predicted boxes={result['predicted_box_count']}"
        )
    print(f"Demo images and summary saved to: {destination}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("phase1", "phase2", "both"),
        default="both",
        help="Run Phase 1 CAM, Phase 2 Grad-CAM++, or both (default: both).",
    )
    parser.add_argument(
        "--image-id",
        help=(
            "Pneumonia-positive image ID in the existing test split. "
            "Defaults to the first test-positive ID."
        ),
    )
    args = parser.parse_args()
    run_demo(phase=args.phase, image_id=args.image_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
