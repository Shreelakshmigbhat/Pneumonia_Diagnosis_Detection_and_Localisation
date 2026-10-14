"""Train and evaluate the paper-inspired 10-layer CNN on the fixed split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import PneumoniaDataset, SPLIT_METADATA
from src.data.preprocessing import DEFAULT_STATS_PATH, load_normalization_stats
from src.models.cnn import INPUT_SIZE, Custom10LayerCNN, count_parameters

CHECKPOINT_PATH = Path("models") / "cnn_best.pth"
RESULT_PATH = Path("results") / "cnn_results.json"
LR_RESULT_PATH = Path("results") / "logistic_regression.json"
SVM_RESULT_PATH = Path("results") / "svm.json"
RF_RESULT_PATH = Path("results") / "random_forest.json"
SEED = 42
LEARNING_RATE = 0.0001
EPOCHS = 20
BATCH_SIZE = 32


@dataclass(frozen=True)
class CNNTrainingConfig:
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    batch_size: int = BATCH_SIZE
    seed: int = SEED
    num_workers: int = 2
    pin_memory: bool = False
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.epochs != EPOCHS:
            raise ValueError("Reference reproduction requires exactly 20 epochs")
        if self.learning_rate != LEARNING_RATE:
            raise ValueError("Reference reproduction requires learning rate 0.0001")
        if self.seed != SEED:
            raise ValueError("Reference reproduction requires seed 42")
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("Batch size must be positive and workers non-negative")


class TorchPneumoniaDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Adapt the shared lazy NumPy dataset to a PyTorch Dataset."""

    def __init__(self, dataset: PneumoniaDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset[index]
        image = torch.from_numpy(sample["image"]).unsqueeze(0)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return image, label


def _set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def _split_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as split_file:
        for chunk in iter(lambda: split_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_counts(dataset: PneumoniaDataset) -> dict[str, int]:
    return {
        "healthy": sum(record["label"] == "0" for record in dataset.records),
        "pneumonia": sum(record["label"] == "1" for record in dataset.records),
    }


def _loader(
    dataset: PneumoniaDataset,
    config: CNNTrainingConfig,
    *,
    shuffle: bool,
    seed_offset: int = 0,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(config.seed + seed_offset)
    return DataLoader(
        TorchPneumoniaDataset(dataset),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        generator=generator,
        drop_last=False,
    )


def _epoch(
    model: Custom10LayerCNN,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    total = 0

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type != "cpu")
        labels = labels.to(device, non_blocking=device.type != "cpu")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        loss_sum += float(loss.detach()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += batch_size
        if batch_index % 100 == 0:
            print(
                f"  {'train' if training else 'validation'} batches: "
                f"{batch_index:,}; images: {total:,}/{len(loader.dataset):,}",
                flush=True,
            )
    return loss_sum / total, correct / total


def _evaluate_predictions(
    model: Custom10LayerCNN,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=device.type != "cpu"))
            all_labels.extend(labels.tolist())
            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
    accuracy = sum(
        actual == predicted
        for actual, predicted in zip(all_labels, all_predictions)
    ) / len(all_labels)
    return accuracy, all_labels, all_predictions



def _confusion_metrics(
    labels: list[int], predictions: list[int]
) -> dict[str, Any]:
    if len(labels) != len(predictions):
        raise ValueError("Labels and predictions must have equal lengths")
    if not labels:
        raise ValueError("Cannot evaluate an empty set of predictions")

    tn = sum(
        actual == 0 and predicted == 0
        for actual, predicted in zip(labels, predictions)
    )
    fp = sum(
        actual == 0 and predicted == 1
        for actual, predicted in zip(labels, predictions)
    )
    fn = sum(
        actual == 1 and predicted == 0
        for actual, predicted in zip(labels, predictions)
    )
    tp = sum(
        actual == 1 and predicted == 1
        for actual, predicted in zip(labels, predictions)
    )

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1_score = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity
        else 0.0
    )
    balanced_accuracy = (sensitivity + specificity) / 2

    return {
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1_score,
        "balanced_accuracy": balanced_accuracy,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }



def _baseline_comparison_guard(
    root: Path, split_digest: str, normalization: dict[str, Any]
) -> None:
    for relative_path in (LR_RESULT_PATH, SVM_RESULT_PATH, RF_RESULT_PATH):
        path = root / relative_path
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        dataset = result["dataset"]
        if dataset["split_metadata_sha256"] != split_digest:
            raise ValueError(f"Existing baseline uses a different split: {path}")
        if dataset["normalization"] != normalization:
            raise ValueError(f"Existing baseline uses different preprocessing: {path}")


def run_training(config: CNNTrainingConfig | None = None) -> dict[str, Any]:
    config = config or CNNTrainingConfig()
    root = PROJECT_ROOT
    _set_seed(config.seed, config.deterministic)
    device = _device()
    if device.type != "cpu" and not config.pin_memory:
        config = replace(config, pin_memory=True)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    print(f"Device: {device}; PyTorch: {torch.__version__}; batch size: {config.batch_size}")

    split_path = root / SPLIT_METADATA
    split_digest = _split_hash(split_path)
    stats = load_normalization_stats(root / DEFAULT_STATS_PATH)
    if stats.source_split != "train" or stats.image_size != INPUT_SIZE:
        raise ValueError("Expected shared 128x128 normalization statistics from train")
    normalization_record = {
        "method": "training-split global pixel mean and standard deviation",
        "stats_path": DEFAULT_STATS_PATH.as_posix(),
        "mean": stats.mean,
        "std": stats.std,
        "source_split": stats.source_split,
        "fit_image_count": 10405,
    }
    _baseline_comparison_guard(root, split_digest, normalization_record)

    datasets = {
        split: PneumoniaDataset(root, split, stats, SPLIT_METADATA)
        for split in ("train", "validation", "test")
    }
    loaders = {
        "train": _loader(datasets["train"], config, shuffle=True),
        "validation": _loader(datasets["validation"], config, shuffle=False, seed_offset=1),
    }

    model = Custom10LayerCNN().to(device)
    cam_report = model.verify_cam_compatibility()
    print(f"Convolutional layers: {sum(isinstance(m, nn.Conv2d) for m in model.modules())}")
    print(f"Fully connected layers: {sum(isinstance(m, nn.Linear) for m in model.modules())}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"CAM compatibility: {cam_report}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict[str, float | int]] = []
    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    best_epoch = 0
    checkpoint_path = root / CHECKPOINT_PATH
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}", flush=True)
        train_loss, train_accuracy = _epoch(
            model, loaders["train"], criterion, device, optimizer
        )
        validation_loss, validation_accuracy = _epoch(
            model, loaders["validation"], criterion, device, None
        )
        epoch_record = {
            "epoch": epoch,
            "training_loss": train_loss,
            "validation_loss": validation_loss,
            "training_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
        }
        history.append(epoch_record)
        print(
            f"  loss train={train_loss:.5f} val={validation_loss:.5f}; "
            f"accuracy train={train_accuracy:.4%} val={validation_accuracy:.4%}",
            flush=True,
        )

        if (
            validation_accuracy > best_validation_accuracy
            or (
                validation_accuracy == best_validation_accuracy
                and validation_loss < best_validation_loss
            )
        ):
            best_validation_accuracy = validation_accuracy
            best_validation_loss = validation_loss
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "architecture": model.architecture_config(),
                    "training_config": asdict(config),
                    "epoch": epoch,
                    "validation_accuracy": validation_accuracy,
                    "validation_loss": validation_loss,
                    "split_metadata_sha256": split_digest,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["split_metadata_sha256"] != split_digest:
        raise ValueError("Saved checkpoint split hash does not match current split")
    model.load_state_dict(checkpoint["state_dict"])

    train_eval_loader = _loader(datasets["train"], config, shuffle=False, seed_offset=2)
    train_accuracy, _, _ = _evaluate_predictions(model, train_eval_loader, device)
    validation_accuracy, _, _ = _evaluate_predictions(
        model, loaders["validation"], device
    )
    test_loader = _loader(datasets["test"], config, shuffle=False, seed_offset=3)
    test_accuracy, test_labels, test_predictions = _evaluate_predictions(
        model, test_loader, device
    )
    test_metrics = _confusion_metrics(test_labels, test_predictions)

    result: dict[str, Any] = {
        "model": "custom 10-layer CNN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "optimizer": "Adam",
        "batch_size": config.batch_size,
        "seed": config.seed,
        "device": str(device),
        "dataset": {
            "split_metadata": SPLIT_METADATA.as_posix(),
            "split_metadata_sha256": split_digest,
            "split_counts": {
                split: {
                    "total": len(dataset),
                    **_class_counts(dataset),
                }
                for split, dataset in datasets.items()
            },
            "label_mapping": {"0": "Healthy/Normal", "1": "Pneumonia/Lung Opacity"},
            "image_shape": [1, *INPUT_SIZE],
            "feature_representation": "128x128 normalized grayscale input tensor",
            "normalization": normalization_record,
        },
        "architecture": model.architecture_config(),
        "parameter_count": count_parameters(model),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "best_validation_loss": best_validation_loss,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        **test_metrics,
        "history": history,
        "reference": {
            "train_accuracy": 0.9307,
            "test_accuracy": 0.9247,
        },
        "reference_cohort_match": False,
        "cam_compatibility": {
            **cam_report,
            "final_feature_maps_available": True,
            "single_final_fc_layer": True,
            "cam_implemented": False,
        },
        "checkpoint": CHECKPOINT_PATH.as_posix(),
    }
    result_path = root / RESULT_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Best epoch: {best_epoch}")
    print(f"Training accuracy: {train_accuracy:.4%}")
    print(f"Validation accuracy: {validation_accuracy:.4%}")
    print(f"Test accuracy: {test_accuracy:.4%}")
    print(f"Sensitivity: {test_metrics['sensitivity']:.4%}")
    print(f"Specificity: {test_metrics['specificity']:.4%}")
    print(f"Saved checkpoint: {checkpoint_path.relative_to(root)}")
    print(f"Saved results: {result_path.relative_to(root)}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the custom 10-layer CNN.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(argv)
    run_training(
        CNNTrainingConfig(batch_size=args.batch_size, num_workers=args.num_workers)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
