"""Training and validation utilities for the boundary-aware U-Net."""

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from pancancer_nuclei.data.targets import create_boundary_target
from pancancer_nuclei.evaluation.semantic import (
    DEFAULT_CLASS_NAMES,
    calculate_semantic_metrics,
    semantic_confusion_matrix,
)

SPATIAL_CLASS_NAMES = ("background", "interior", "boundary")


def boundary_segmentation_collate(
    samples: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Combine samples and derive spatial targets after augmentation."""
    if not samples:
        raise ValueError("Cannot create an empty batch.")

    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "semantic_mask": torch.stack([sample["semantic_mask"] for sample in samples]),
        "spatial_mask": torch.stack(
            [
                torch.as_tensor(
                    create_boundary_target(sample["instance_masks"]),
                    dtype=torch.long,
                )
                for sample in samples
            ]
        ),
    }


def train_boundary_epoch(
    model: nn.Module,
    data_loader: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Any | None = None,
) -> dict[str, float]:
    """Train for one epoch and return mean total and component losses."""
    model.train()

    totals = {"loss": 0.0, "semantic_loss": 0.0, "spatial_loss": 0.0}
    number_of_examples = 0

    for batch in data_loader:
        images = batch["image"].to(device)
        semantic_targets = batch["semantic_mask"].to(device)
        spatial_targets = batch["spatial_mask"].to(device)
        batch_size = images.shape[0]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=scaler is not None,
        ):
            semantic_logits, spatial_logits = model(images)
            result = criterion(
                semantic_logits,
                spatial_logits,
                semantic_targets,
                spatial_targets,
            )

        if scaler is None:
            result.total_loss.backward()
            optimizer.step()
        else:
            scaler.scale(result.total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        totals["loss"] += result.total_loss.item() * batch_size
        totals["semantic_loss"] += result.semantic_loss.item() * batch_size
        totals["spatial_loss"] += result.spatial_loss.item() * batch_size
        number_of_examples += batch_size

    if number_of_examples == 0:
        raise ValueError("The training data loader was empty.")

    return {name: value / number_of_examples for name, value in totals.items()}


@torch.no_grad()
def evaluate_boundary_model(
    model: nn.Module,
    data_loader: Any,
    criterion: nn.Module,
    device: torch.device,
    semantic_class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    spatial_class_names: Sequence[str] = SPATIAL_CLASS_NAMES,
    ignore_label: int = 255,
) -> dict[str, Any]:
    """Evaluate both heads and return losses and confusion-based metrics."""
    model.eval()

    totals = {"loss": 0.0, "semantic_loss": 0.0, "spatial_loss": 0.0}
    number_of_examples = 0
    semantic_confusion = torch.zeros(6, 6, dtype=torch.int64)
    spatial_confusion = torch.zeros(3, 3, dtype=torch.int64)

    for batch in data_loader:
        images = batch["image"].to(device)
        semantic_targets = batch["semantic_mask"].to(device)
        spatial_targets = batch["spatial_mask"].to(device)
        batch_size = images.shape[0]

        semantic_logits, spatial_logits = model(images)
        result = criterion(
            semantic_logits,
            spatial_logits,
            semantic_targets,
            spatial_targets,
        )

        totals["loss"] += result.total_loss.item() * batch_size
        totals["semantic_loss"] += result.semantic_loss.item() * batch_size
        totals["spatial_loss"] += result.spatial_loss.item() * batch_size
        number_of_examples += batch_size

        semantic_confusion += semantic_confusion_matrix(
            semantic_logits.detach().cpu(),
            semantic_targets.detach().cpu(),
            number_of_classes=6,
            ignore_label=ignore_label,
        )
        spatial_confusion += semantic_confusion_matrix(
            spatial_logits.detach().cpu(),
            spatial_targets.detach().cpu(),
            number_of_classes=3,
            ignore_label=ignore_label,
        )

    if number_of_examples == 0:
        raise ValueError("The validation data loader was empty.")

    semantic_metrics = calculate_semantic_metrics(
        semantic_confusion,
        class_names=semantic_class_names,
    )
    spatial_metrics = calculate_semantic_metrics(
        spatial_confusion,
        class_names=spatial_class_names,
    )

    semantic_dice = semantic_metrics["macro_foreground_dice"] or 0.0
    spatial_dice = spatial_metrics["macro_foreground_dice"] or 0.0

    return {
        "loss": totals["loss"] / number_of_examples,
        "semantic_loss": totals["semantic_loss"] / number_of_examples,
        "spatial_loss": totals["spatial_loss"] / number_of_examples,
        "semantic": {
            **semantic_metrics,
            "confusion_matrix": semantic_confusion.tolist(),
        },
        "spatial": {
            **spatial_metrics,
            "confusion_matrix": spatial_confusion.tolist(),
        },
        "selection_score": (semantic_dice + spatial_dice) / 2.0,
    }
