"""Build presentation figures from the fixed test split and frozen CNN."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "presentation_assets"
THRESHOLD = 0.5
ALPHA = 0.4
OUTPUT_SIZE = 1024
PNEUMONIA = 1
TARGET_METRICS = {
    "accuracy": 0.880215,
    "precision": 0.920477,
    "recall": 0.770383,
    "F1": 0.838768,
}
CASE_FILENAMES = (
    "input.png",
    "phase1_cam.png",
    "phase2_gradcam.png",
)

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
from src.localization.improved_cam import compute_gradcam_plus_plus  # noqa: E402
from src.localization.iou import image_iou  # noqa: E402
from src.localization.pipeline import (  # noqa: E402
    _dicom_dimensions,
    load_cnn_checkpoint,
)
from src.models.cnn import INPUT_SIZE  # noqa: E402


def _read_split(root: Path) -> list[dict[str, str]]:
    path = root / "metadata" / "split_metadata.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
    if not records or not {"image_id", "image_path", "label", "split", "x", "y", "width", "height"}.issubset(records[0]):
        raise ValueError(f"Missing or invalid fixed split metadata: {path}")
    test_records = [row for row in records if row["split"] == "test"]
    if len(test_records) != 1486:
        raise ValueError(f"Expected 1,486 fixed test rows, found {len(test_records)}")
    if any(row["label"] not in {"0", "1"} for row in test_records):
        raise ValueError("Unexpected label in fixed test split")
    return test_records


def _image_path(root: Path, record: dict[str, str]) -> Path:
    path = (root / record["image_path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise FileNotFoundError(f"Invalid or missing test image: {path}")
    return path


def _full_range_gray(pixels: np.ndarray) -> Image.Image:
    values = np.asarray(pixels, dtype=np.float32)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected a finite, non-empty grayscale image")
    low, high = float(values.min()), float(values.max())
    if high <= low:
        raise ValueError("Cannot window a constant-valued radiograph")
    array = np.rint((values - low) * (255.0 / (high - low))).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def _fit_square(image: Image.Image) -> tuple[Image.Image, float, float, int, int]:
    width, height = image.size
    scale = min(OUTPUT_SIZE / width, OUTPUT_SIZE / height)
    fitted_width = max(1, round(width * scale))
    fitted_height = max(1, round(height * scale))
    resized = image.resize(
        (fitted_width, fitted_height), resample=Image.Resampling.LANCZOS
    )
    left = (OUTPUT_SIZE - fitted_width) // 2
    top = (OUTPUT_SIZE - fitted_height) // 2
    square = Image.new("RGB", (OUTPUT_SIZE, OUTPUT_SIZE), "black")
    square.paste(resized.convert("RGB"), (left, top))
    return square, fitted_width / width, fitted_height / height, left, top


def _jet(heatmap: np.ndarray, size: tuple[int, int]) -> Image.Image:
    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite, two-dimensional activation map")
    image = Image.fromarray(values.clip(0, 1), mode="F")
    resized = np.asarray(
        image.resize(size, resample=Image.Resampling.BILINEAR), dtype=np.float32
    ).clip(0, 1)
    red = np.clip(1.5 - np.abs(4.0 * resized - 3.0), 0, 1)
    green = np.clip(1.5 - np.abs(4.0 * resized - 2.0), 0, 1)
    blue = np.clip(1.5 - np.abs(4.0 * resized - 1.0), 0, 1)
    rgb = np.rint(np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _draw_overlay(
    gray: Image.Image,
    heatmap: np.ndarray,
    ground_truth: list[tuple[float, float, float, float]],
    predictions: list[tuple[float, float, float, float]],
) -> Image.Image:
    canvas, scale_x, scale_y, left, top = _fit_square(gray)
    resized_heatmap = _jet(heatmap, gray.size)
    fitted_width = round(gray.width * scale_x)
    fitted_height = round(gray.height * scale_y)
    resized_heatmap = resized_heatmap.resize(
        (fitted_width, fitted_height), resample=Image.Resampling.BILINEAR
    )
    heatmap_square = Image.new("RGB", canvas.size, "black")
    heatmap_square.paste(resized_heatmap, (left, top))
    canvas = Image.blend(canvas, heatmap_square, alpha=ALPHA)
    draw = ImageDraw.Draw(canvas)
    for boxes, color in ((ground_truth, (0, 200, 83)), (predictions, (255, 109, 0))):
        for x, y, width, height in boxes:
            draw.rectangle(
                (
                    left + x * scale_x,
                    top + y * scale_y,
                    left + (x + width) * scale_x,
                    top + (y + height) * scale_y,
                ),
                outline=color,
                width=4,
            )
    return canvas


def _boxes_for_map(
    cam: np.ndarray, image_size: tuple[int, int]
) -> list[tuple[float, float, float, float]]:
    clusters = find_clusters(activated_mask(cam, THRESHOLD))
    return clusters_to_boxes(
        clusters,
        map_size=cam.shape,
        image_size=image_size,
        apply_two_sigma=True,
    )


def _boxes_from_record(
    record: dict[str, str],
) -> list[tuple[float, float, float, float]]:
    coordinates = [json.loads(record[name]) for name in ("x", "y", "width", "height")]
    if len({len(values) for values in coordinates}) != 1:
        raise ValueError(f"Box coordinate array lengths differ for {record['image_id']}")
    return [
        tuple(float(coordinates[axis][index]) for axis in range(4))
        for index in range(len(coordinates[0]))
    ]


def _load_indexed_examples(root: Path) -> list[dict[str, Any]]:
    index_path = root / "results" / "phase2_error_analysis" / "examples.csv"
    with index_path.open(newline="", encoding="utf-8-sig") as stream:
        indexed = list(csv.DictReader(stream))
    if len(indexed) != 12:
        raise ValueError(f"Expected 12 saved Stage 12 examples; found {len(indexed)}")
    for row in indexed:
        visualization = root / row["visualization"]
        if not visualization.is_file():
            raise FileNotFoundError(f"Missing indexed Stage 12 example: {visualization}")

    with (root / "results" / "cam_results.json").open(encoding="utf-8") as stream:
        phase1 = json.load(stream)
    phase1_by_id = {row["image_id"]: row for row in phase1["image_results"]}
    candidates: list[dict[str, Any]] = []
    for row in indexed:
        image_id = row["image_id"]
        if image_id not in phase1_by_id:
            raise ValueError(f"Phase 1 result missing indexed image {image_id}")
        if not math.isclose(
            float(row["iou"]),
            float(phase1_by_id[image_id]["iou"]),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError(f"Stage 12 IoU does not match saved Phase 1 IoU for {image_id}")
        candidates.append(
            {
                "image_id": image_id,
                "phase1_iou_saved": float(phase1_by_id[image_id]["iou"]),
                "phase1_count_saved": int(phase1_by_id[image_id]["predicted_box_count"]),
                "stage12_example": row,
            }
        )
    return candidates


def _phase2_example_index(root: Path) -> list[dict[str, str]]:
    path = root / "results" / "phase2_improved_examples" / "examples.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        examples = list(csv.DictReader(stream))
    if not examples:
        raise ValueError(f"Saved Phase 2 example index is empty: {path}")
    return examples


def _select_cases(
    candidates: list[dict[str, Any]], median_iou: float
) -> list[dict[str, Any]]:
    success = max(candidates, key=lambda row: row["phase2_iou"])
    partial = min(candidates, key=lambda row: abs(row["phase2_iou"] - median_iou))
    failure = min(candidates, key=lambda row: row["phase2_iou"])
    if len({success["image_id"], partial["image_id"], failure["image_id"]}) != 3:
        raise ValueError("The 12 saved examples do not yield three distinct IoU cases")
    for name, row in (
        ("A_success", success),
        ("B_partial", partial),
        ("C_failure", failure),
    ):
        row["case"] = name
    return [success, partial, failure]


def _classification_metrics(
    root: Path,
    records: list[dict[str, str]],
    model: torch.nn.Module,
    stats: Any,
    device: torch.device,
) -> tuple[dict[str, Any], list[float], list[int], list[int]]:
    labels = [int(row["label"]) for row in records]
    predictions: list[int] = []
    positive_scores: list[float] = []
    batch_size = 32
    model.eval()
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        batch = np.stack(
            [
                preprocess_image(_image_path(root, row), stats)
                for row in batch_records
            ]
        )
        images = torch.from_numpy(batch).unsqueeze(1).to(device)
        with torch.inference_mode():
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
        predictions.extend(probabilities.argmax(dim=1).cpu().tolist())
        positive_scores.extend(probabilities[:, PNEUMONIA].cpu().tolist())
        done = min(start + len(batch_records), len(records))
        if done == len(records) or done % 256 == 0:
            print(f"Classification inference: {done}/{len(records)}")

    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    metrics = {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(labels, positive_scores)),
    }
    comparison_path = root / "results" / "final_model_comparison.csv"
    with comparison_path.open(newline="", encoding="utf-8-sig") as stream:
        cnn_row = next(row for row in csv.DictReader(stream) if row["Model"] == "CNN")
    for metric, csv_column in (
        ("accuracy", "Test Accuracy"),
        ("precision", "Test Precision"),
        ("recall", "Test Recall"),
        ("F1", "Test F1"),
    ):
        saved_value = float(cnn_row[csv_column])
        if not math.isclose(metrics[metric], saved_value, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(
                f"Classification sanity check failed: {metric}={metrics[metric]:.10f}, "
                f"saved CNN value={saved_value:.10f}"
            )
    for metric, expected in TARGET_METRICS.items():
        if not math.isclose(metrics[metric], expected, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(
                f"Classification sanity check failed: {metric}={metrics[metric]:.10f}, "
                f"requested reference={expected:.6f}"
            )
    return metrics, positive_scores, labels, predictions


def _font(size: int) -> ImageFont.FreeTypeFont:
    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    if not font_path.is_file():
        raise FileNotFoundError(f"Required presentation font is missing: {font_path}")
    return ImageFont.truetype(str(font_path), size)


def _draw_confusion_matrix(metrics: dict[str, Any]) -> Image.Image:
    image = Image.new("RGBA", (1200, 1120), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = _font(40)
    count_font = _font(58)
    small_font = _font(30)
    x0, y0, cell = 330, 255, 380
    values = ((metrics["TN"], metrics["FP"]), (metrics["FN"], metrics["TP"]))
    maximum = max(max(row) for row in values)
    for row in range(2):
        for column in range(2):
            x, y = x0 + column * cell, y0 + row * cell
            intensity = values[row][column] / maximum if maximum else 0
            shade = int(238 - 95 * intensity)
            draw.rectangle(
                (x, y, x + cell, y + cell),
                fill=(shade, shade + 6, 255, 255),
                outline=(255, 255, 255, 255),
                width=6,
            )
            draw.text(
                (x + cell / 2, y + cell / 2),
                str(values[row][column]),
                fill=(20, 35, 55, 255),
                font=count_font,
                anchor="mm",
            )
    draw.text((x0 + cell, 105), "Predicted class", fill=(25, 35, 50, 255), font=font, anchor="mm")
    draw.text((x0 + cell / 2, 190), "Healthy", fill=(25, 35, 50, 255), font=small_font, anchor="mm")
    draw.text((x0 + 1.5 * cell, 190), "Pneumonia", fill=(25, 35, 50, 255), font=small_font, anchor="mm")
    draw.text((160, y0 + cell), "True class", fill=(25, 35, 50, 255), font=font, anchor="mm")
    draw.text((x0 - 25, y0 + cell / 2), "Healthy", fill=(25, 35, 50, 255), font=small_font, anchor="rm")
    draw.text((x0 - 25, y0 + 1.5 * cell), "Pneumonia", fill=(25, 35, 50, 255), font=small_font, anchor="rm")
    return image


def _draw_roc(scores: list[float], labels: list[int], auc: float) -> Image.Image:
    false_positive, true_positive, _ = roc_curve(labels, scores)
    image = Image.new("RGBA", (1200, 950), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = _font(34)
    small_font = _font(26)
    left, right, top, bottom = 180, 1080, 150, 780
    for tick in range(6):
        fraction = tick / 5
        x = round(left + fraction * (right - left))
        y = round(bottom - fraction * (bottom - top))
        draw.line((x, top, x, bottom), fill=(210, 215, 222, 210), width=2)
        draw.line((left, y, right, y), fill=(210, 215, 222, 210), width=2)
        label = f"{fraction:.1f}"
        draw.text((x, bottom + 26), label, fill=(40, 48, 58, 255), font=small_font, anchor="mm")
        draw.text((left - 36, y), label, fill=(40, 48, 58, 255), font=small_font, anchor="rm")
    draw.line((left, bottom, right, top), fill=(135, 143, 153, 255), width=3)
    points = [
        (
            round(left + float(fpr) * (right - left)),
            round(bottom - float(tpr) * (bottom - top)),
        )
        for fpr, tpr in zip(false_positive, true_positive)
    ]
    draw.line(points, fill=(36, 104, 180, 255), width=6, joint="curve")
    draw.text((630, 65), f"CNN ROC  (AUC = {auc:.4f})", fill=(25, 35, 50, 255), font=font, anchor="mm")
    draw.text(((left + right) / 2, 885), "False-positive rate", fill=(35, 43, 52, 255), font=font, anchor="mm")
    vertical_label = Image.new("RGBA", (380, 48), (0, 0, 0, 0))
    ImageDraw.Draw(vertical_label).text(
        (190, 24),
        "True-positive rate",
        fill=(35, 43, 52, 255),
        font=font,
        anchor="mm",
    )
    vertical_label = vertical_label.rotate(90, expand=True)
    image.alpha_composite(vertical_label, dest=(38, (image.height - vertical_label.height) // 2))
    return image


def _preview_normals(records: list[dict[str, str]], root: Path) -> None:
    normals = sorted((row for row in records if row["label"] == "0"), key=lambda row: row["image_id"])
    positions = np.linspace(0, len(normals) - 1, 16, dtype=int)
    selected = [normals[int(index)] for index in positions]
    cell, label_height = 280, 28
    sheet = Image.new("RGB", (cell * 4, (cell + label_height) * 4), "black")
    draw = ImageDraw.Draw(sheet)
    label_font = ImageFont.load_default()
    for index, row in enumerate(selected):
        gray = _full_range_gray(read_dicom_pixels(_image_path(root, row))[0])
        square = gray.copy()
        square.thumbnail((cell, cell), Image.Resampling.LANCZOS)
        x, y = (index % 4) * cell, (index // 4) * (cell + label_height)
        sheet.paste(square, (x + (cell - square.width) // 2, y + (cell - square.height) // 2))
        draw.text((x + 4, y + cell), row["image_id"], font=label_font, fill="white")
    ASSET_DIR.mkdir(exist_ok=True)
    preview_path = ASSET_DIR / "_normal_screening.png"
    if preview_path.exists():
        raise FileExistsError(f"Refusing to overwrite {preview_path}")
    sheet.save(preview_path)
    print(f"Inspect normal candidates in {preview_path}")


def _write_readme(
    path: Path,
    normal_id: str,
    cases: list[dict[str, Any]],
    checkpoint_epoch: int,
    metrics: dict[str, Any],
) -> None:
    files = [
        "make_assets.py",
        "clean_normal.png",
        "clean_pneumonia.png",
        *(f"case_{letter}_{name}" for letter in "ABC" for name in CASE_FILENAMES),
        "cases_summary.csv",
        "confusion_matrix.png",
        "roc_curve.png",
        "classification_check.json",
        "README.md",
    ]
    case_lines = "\n".join(
        f"- {item['case']}: `{item['image_id']}`; Phase 1 IoU "
        f"{item['phase1_iou']:.6f}; Phase 2 IoU {item['phase2_iou']:.6f}."
        for item in cases
    )
    contents = f"""# Presentation assets

## Files

{chr(10).join(f"- `{name}`" for name in files)}

## Image IDs and case selection

- Clean Normal image (label `Normal`, test split): `{normal_id}`.
- Clean Pneumonia image (label `Lung Opacity`, test split): `{cases[0]['image_id']}`.
{case_lines}

The 12 candidate IDs came from the existing Stage 12 failure-analysis index in `results/phase2_error_analysis/examples.csv`; its indexed IoUs are Phase 1 baseline values, verified against `results/cam_results.json`. Phase 2 maps and per-image IoUs were then generated for those 12 candidates with the saved method. Cases are the highest Phase 2 IoU, the candidate nearest the saved Phase 2 full-test median ({float(json.loads((ROOT / 'results' / 'phase2_improved_cam_results.json').read_text(encoding='utf-8'))['median_iou']):.6f}), and the lowest Phase 2 IoU. The Grad-CAM++ implementation was verified against a saved Stage 13 example before this selection. Phase 1 maps and both sets of boxes were produced only for the three selected test images, using the unchanged fixed split and epoch-{checkpoint_epoch} checkpoint.

## Production and assumptions

- Source pixels were decoded with `src.data.preprocessing.read_dicom_pixels`; the clean PNGs are independently min-max windowed to the full stored pixel range (after the DICOM MONOCHROME1 handling), converted to 8-bit grayscale, and resized without added overlays or text. Any orientation/acquisition markers already encoded in the source pixels remain. The clean files preserve aspect ratio with their long side at 1024 px.
- CNN inputs use the existing train-fitted preprocessing function and statistics. Phase 1 maps use `src.localization.cam.compute_cam`; Phase 2 maps use `src.localization.improved_cam.compute_gradcam_plus_plus`. Both use class 1, inclusive threshold 0.50, iterative 8-connected clustering, and the existing 2-SD region filter and box conversion.
- Ground-truth boxes come from each test row in the unchanged `metadata/split_metadata.csv`. `src.localization.iou.image_iou` is used for greedy one-to-one IoU matching. DICOM-native box coordinates are transformed through one shared aspect-preserving square fit for each case image and both overlays.
- Result overlays use a JET heatmap at 0.40 opacity, ground-truth green `#00C853`, predicted orange `#FF6D00`, and 4 px outlines. Square case images are padded, not cropped, so all layers share the same scale and placement.
- Classification was inference only on all 1,486 test images. Metrics passed the sanity check against `results/final_model_comparison.csv`: accuracy {metrics['accuracy']:.9f}; precision {metrics['precision']:.9f}; recall {metrics['recall']:.9f}; F1 {metrics['F1']:.9f}. The ROC-AUC and confusion counts in the JSON were computed from this inference.
- No model was retrained, no split was generated, and files in `results/`, `metadata/`, and `notebooks/` were not modified. No external packages were installed.

"""
    path.write_text(contents, encoding="utf-8")


def _build(normal_id: str) -> None:
    records = _read_split(ROOT)
    ASSET_DIR.mkdir(exist_ok=True)
    forbidden = [ASSET_DIR / name for name in (
        "clean_normal.png", "clean_pneumonia.png", "cases_summary.csv",
        "confusion_matrix.png", "roc_curve.png", "classification_check.json", "README.md",
        *(f"case_{letter}_{filename}" for letter in "ABC" for filename in CASE_FILENAMES),
    )]
    existing = [path for path in forbidden if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite presentation assets: {existing}")
    by_id = {record["image_id"]: record for record in records}
    normal = by_id.get(normal_id)
    if normal is None or normal["label"] != "0":
        raise ValueError(f"{normal_id} is not a Normal image in the fixed test split")

    model, device, checkpoint_info = load_cnn_checkpoint(ROOT)
    if checkpoint_info["checkpoint_epoch"] != 9:
        raise ValueError(f"Expected epoch-9 checkpoint, found {checkpoint_info['checkpoint_epoch']}")
    stats = load_normalization_stats(ROOT / DEFAULT_STATS_PATH)
    if stats.source_split != "train" or stats.image_size != INPUT_SIZE:
        raise ValueError("Expected the saved train-only 128x128 preprocessing statistics")

    print("Running frozen-checkpoint inference on the fixed test split.")
    metrics, scores, labels, predictions = _classification_metrics(
        ROOT, records, model, stats, device
    )
    print("Classification metrics passed saved-result sanity checks.")
    candidates = _load_indexed_examples(ROOT)
    positive_by_id = {record["image_id"]: record for record in records}
    phase2_examples = _phase2_example_index(ROOT)

    # Cross-check this generator against an independently saved Stage 13 case.
    validation_example = phase2_examples[0]
    validation_record = positive_by_id.get(validation_example["image_id"])
    if validation_record is None:
        raise ValueError("Saved Phase 2 example is not in the fixed positive test split")
    validation_path = _image_path(ROOT, validation_record)
    validation_input = torch.from_numpy(
        preprocess_image(validation_path, stats)
    ).unsqueeze(0).unsqueeze(0).to(device)
    validation_map = compute_gradcam_plus_plus(
        model,
        validation_input,
        class_index=PNEUMONIA,
        output_size=INPUT_SIZE,
        epsilon=1e-8,
    )[0].cpu().numpy()
    validation_boxes = _boxes_for_map(
        validation_map, _dicom_dimensions(validation_path)
    )
    validation_iou = image_iou(
        validation_boxes, _boxes_from_record(validation_record)
    )
    if not math.isclose(
        validation_iou,
        float(validation_example["improved_iou"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError(
            "Grad-CAM++ verification against the saved Phase 2 example failed: "
            f"computed {validation_iou:.12f}, saved "
            f"{float(validation_example['improved_iou']):.12f}"
        )
    print("Grad-CAM++ generation matches the saved Stage 13 example.")

    phase2_cache: dict[str, np.ndarray] = {}
    with (ROOT / "results" / "phase2_improved_cam_results.json").open(
        encoding="utf-8"
    ) as stream:
        saved_phase2 = json.load(stream)
    for candidate in candidates:
        record = positive_by_id[candidate["image_id"]]
        path = _image_path(ROOT, record)
        image = torch.from_numpy(preprocess_image(path, stats)).unsqueeze(0).unsqueeze(0).to(device)
        phase2_map = compute_gradcam_plus_plus(
            model,
            image,
            class_index=PNEUMONIA,
            output_size=INPUT_SIZE,
            epsilon=1e-8,
        )[0].cpu().numpy()
        boxes = _boxes_for_map(phase2_map, _dicom_dimensions(path))
        candidate["phase2_iou"] = image_iou(boxes, _boxes_from_record(record))
        candidate["phase2_count"] = len(boxes)
        phase2_cache[candidate["image_id"]] = phase2_map
    selected = _select_cases(candidates, float(saved_phase2["median_iou"]))

    # Build every image in memory only after the classification guard succeeds.
    normal_pixels, _ = read_dicom_pixels(_image_path(ROOT, normal))
    clean_normal = _full_range_gray(normal_pixels)
    clean_normal.thumbnail((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS)

    case_outputs: list[dict[str, Any]] = []
    case_pngs: list[tuple[str, Image.Image]] = []
    summary_rows: list[dict[str, Any]] = []
    phase1_path = ROOT / "results" / "cam_results.json"
    with phase1_path.open(encoding="utf-8") as stream:
        phase1_results = json.load(stream)
    phase1_rows = {row["image_id"]: row for row in phase1_results["image_results"]}
    for item in selected:
        record = positive_by_id[item["image_id"]]
        dicom_path = _image_path(ROOT, record)
        pixels, _ = read_dicom_pixels(dicom_path)
        native_gray = _full_range_gray(pixels)
        image_size = (native_gray.height, native_gray.width)
        image_tensor = torch.from_numpy(preprocess_image(dicom_path, stats)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(image_tensor)
            probability = float(torch.softmax(logits, dim=1)[0, PNEUMONIA].item())
            predicted_class = int(logits.argmax(dim=1).item())
            phase1_map = compute_cam(
                model, image_tensor, class_index=PNEUMONIA, output_size=INPUT_SIZE
            )[0].cpu().numpy()
        phase2_map = phase2_cache[item["image_id"]]
        gt_boxes = _boxes_from_record(record)
        phase1_boxes = _boxes_for_map(phase1_map, image_size)
        phase2_boxes = _boxes_for_map(phase2_map, image_size)
        phase1_iou = image_iou(phase1_boxes, gt_boxes)
        phase2_iou = image_iou(phase2_boxes, gt_boxes)
        saved_phase1 = phase1_rows[item["image_id"]]
        for phase_name, actual, saved in (
            ("Phase 1", phase1_iou, float(saved_phase1["iou"])),
            ("Phase 2", phase2_iou, float(item["phase2_iou"])),
        ):
            if not math.isclose(actual, saved, rel_tol=0.0, abs_tol=1e-10):
                raise RuntimeError(
                    f"{phase_name} IoU mismatch for {item['image_id']}: "
                    f"computed {actual:.12f}, saved {saved:.12f}"
                )
        if len(phase1_boxes) != int(saved_phase1["predicted_box_count"]):
            raise RuntimeError(f"Phase 1 box-count mismatch for {item['image_id']}")
        if len(phase2_boxes) != int(item["phase2_count"]):
            raise RuntimeError(f"Phase 2 box-count mismatch for {item['image_id']}")

        letter = item["case"][0]
        case_pngs.extend(
            (
                (f"case_{letter}_input.png", _fit_square(native_gray)[0]),
                (
                    f"case_{letter}_phase1_cam.png",
                    _draw_overlay(native_gray, phase1_map, gt_boxes, phase1_boxes),
                ),
                (
                    f"case_{letter}_phase2_gradcam.png",
                    _draw_overlay(native_gray, phase2_map, gt_boxes, phase2_boxes),
                ),
            )
        )
        output_row = {
            "case": item["case"],
            "image_id": item["image_id"],
            "true_label": "Lung Opacity",
            "predicted_class": "Pneumonia" if predicted_class == 1 else "Healthy",
            "pneumonia_softmax_confidence": probability,
            "phase1_iou": phase1_iou,
            "phase2_iou": phase2_iou,
            "n_gt_boxes": len(gt_boxes),
            "n_pred_boxes_phase1": len(phase1_boxes),
            "n_pred_boxes_phase2": len(phase2_boxes),
        }
        case_outputs.append(output_row)
        summary_rows.append(output_row)
        print(
            f"{item['case']}: {item['image_id']} "
            f"(Phase 1 IoU {phase1_iou:.6f}, Phase 2 IoU {phase2_iou:.6f})"
        )

    pneumonia_record = positive_by_id[case_outputs[0]["image_id"]]
    pneumonia_pixels, _ = read_dicom_pixels(_image_path(ROOT, pneumonia_record))
    clean_pneumonia = _full_range_gray(pneumonia_pixels)
    clean_pneumonia.thumbnail((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS)
    confusion = _draw_confusion_matrix(metrics)
    roc_image = _draw_roc(scores, labels, metrics["ROC-AUC"])

    summary_path = ASSET_DIR / "cases_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    clean_normal.save(ASSET_DIR / "clean_normal.png")
    clean_pneumonia.save(ASSET_DIR / "clean_pneumonia.png")
    for filename, image in case_pngs:
        image.save(ASSET_DIR / filename)
    confusion.save(ASSET_DIR / "confusion_matrix.png")
    roc_image.save(ASSET_DIR / "roc_curve.png")
    (ASSET_DIR / "classification_check.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_readme(
        ASSET_DIR / "README.md",
        normal_id,
        case_outputs,
        int(checkpoint_info["checkpoint_epoch"]),
        metrics,
    )
    print(f"Clean Normal image ID: {normal_id}")
    print(f"Clean Pneumonia image ID: {pneumonia_record['image_id']}")
    print(f"All presentation assets saved in {ASSET_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview-normals",
        action="store_true",
        help="Write a temporary contact sheet of 16 Normal test images for selection.",
    )
    parser.add_argument(
        "--normal-id",
        help="Image ID of a Normal case from the fixed test split, selected from the preview.",
    )
    args = parser.parse_args()
    if args.preview_normals:
        if args.normal_id:
            parser.error("--normal-id cannot be combined with --preview-normals")
        _preview_normals(_read_split(ROOT), ROOT)
        return 0
    if not args.normal_id:
        parser.error("Supply --normal-id after reviewing the normal-image contact sheet")
    _build(args.normal_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
