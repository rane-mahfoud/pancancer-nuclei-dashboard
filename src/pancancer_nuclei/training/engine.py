"""Reusable training and validation functions."""

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from pancancer_nuclei.evaluation.semantic import (
    DEFAULT_CLASS_NAMES,
    calculate_semantic_metrics,
    semantic_confusion_matrix,
)


def segmentation_collate(
    samples: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Combine fixed-size images and semantic masks into a batch."""
    if not samples:
        raise ValueError("Cannot create an empty batch.")

    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "semantic_mask": torch.stack([sample["semantic_mask"] for sample in samples]),
    }


def train_one_epoch(
    model: nn.Module,
    data_loader: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Any | None = None,
) -> float:
    """Train for one epoch and return mean loss."""
    model.train()

    total_loss = 0.0
    number_of_examples = 0

    for batch in data_loader:
        images = batch["image"].to(device)
        targets = batch["semantic_mask"].to(device)
        batch_size = images.shape[0]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=scaler is not None,
        ):
            predictions = model(images)
            loss = criterion(predictions, targets)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * batch_size
        number_of_examples += batch_size

    if number_of_examples == 0:
        raise ValueError("The training data loader was empty.")

    return total_loss / number_of_examples


@torch.no_grad()
def evaluate_semantic_model(
    model: nn.Module,
    data_loader: Any,
    criterion: nn.Module,
    device: torch.device,
    number_of_classes: int = 6,
    ignore_label: int = 255,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> dict[str, Any]:
    """Evaluate loss and semantic metrics without changing the model."""
    model.eval()

    total_loss = 0.0
    number_of_examples = 0
    total_confusion = torch.zeros(
        number_of_classes,
        number_of_classes,
        dtype=torch.int64,
    )

    for batch in data_loader:
        images = batch["image"].to(device)
        targets = batch["semantic_mask"].to(device)
        batch_size = images.shape[0]

        predictions = model(images)
        loss = criterion(predictions, targets)

        total_loss += loss.item() * batch_size
        number_of_examples += batch_size

        total_confusion += semantic_confusion_matrix(
            predictions.detach().cpu(),
            targets.detach().cpu(),
            number_of_classes=number_of_classes,
            ignore_label=ignore_label,
        )

    if number_of_examples == 0:
        raise ValueError("The validation data loader was empty.")

    metrics = calculate_semantic_metrics(
        total_confusion,
        class_names=class_names,
    )
    metrics["loss"] = total_loss / number_of_examples
    metrics["confusion_matrix"] = total_confusion.tolist()

    return metrics
