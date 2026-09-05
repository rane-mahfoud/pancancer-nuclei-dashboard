"""Train and validate the boundary-aware PanNuke U-Net."""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from pancancer_nuclei.data.pannuke import PanNukeDataset
from pancancer_nuclei.data.transforms import (
    create_training_transforms,
    create_validation_transforms,
)
from pancancer_nuclei.evaluation.semantic import DEFAULT_CLASS_NAMES
from pancancer_nuclei.models.boundary_loss import (
    BoundaryAwareSegmentationLoss,
)
from pancancer_nuclei.models.boundary_unet import BoundaryAwareUNet
from pancancer_nuclei.models.losses import calculate_log_class_weights
from pancancer_nuclei.training.boundary_engine import (
    SPATIAL_CLASS_NAMES,
    boundary_segmentation_collate,
    evaluate_boundary_model,
    train_boundary_epoch,
)

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = "data/raw/huggingface_cache"
SEMANTIC_AUDIT_PATH = Path("reports/fold1_semantic_pixel_audit.json")
SPATIAL_AUDIT_PATH = Path("reports/fold1_boundary_target_audit.json")
SEMANTIC_TASK_WEIGHT = 1.0
SPATIAL_TASK_WEIGHT = 0.5
SEED = 42


def parse_arguments() -> argparse.Namespace:
    """Read command-line training settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def choose_setting(
    supplied_value: int | None,
    smoke_value: int,
    full_value: int,
    smoke_test: bool,
) -> int:
    """Choose a command-line, smoke-test, or full-training value."""
    if supplied_value is not None:
        return supplied_value
    return smoke_value if smoke_test else full_value


def set_random_seeds() -> None:
    """Set random seeds for repeatability."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def limit_dataset(
    dataset: PanNukeDataset,
    limit: int | None,
    seed: int,
) -> PanNukeDataset | Subset:
    """Optionally select a reproducible random subset."""
    if limit is None:
        return dataset
    if limit <= 0:
        raise ValueError("Dataset limits must be positive.")

    generator = torch.Generator().manual_seed(seed)
    number_selected = min(limit, len(dataset))
    selected_indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:number_selected].tolist()
    return Subset(dataset, selected_indices)


def create_data_loaders(
    batch_size: int,
    train_limit: int | None,
    validation_limit: int | None,
) -> tuple[DataLoader, DataLoader]:
    """Create Fold 1 training and Fold 2 validation loaders."""
    fold1 = load_dataset(
        DATASET_NAME,
        revision=REVISION,
        split="fold1",
        cache_dir=CACHE_DIR,
    )
    fold2 = load_dataset(
        DATASET_NAME,
        revision=REVISION,
        split="fold2",
        cache_dir=CACHE_DIR,
    )

    training_dataset = PanNukeDataset(
        fold1,
        transform=create_training_transforms(),
    )
    validation_dataset = PanNukeDataset(
        fold2,
        transform=create_validation_transforms(),
    )
    training_dataset = limit_dataset(training_dataset, train_limit, SEED)
    validation_dataset = limit_dataset(
        validation_dataset,
        validation_limit,
        SEED + 1,
    )

    generator = torch.Generator().manual_seed(SEED)
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=boundary_segmentation_collate,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=boundary_segmentation_collate,
    )
    return training_loader, validation_loader


def load_class_weights(
    audit_path: Path,
    class_names: tuple[str, ...],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Derive reproducible log weights from an audit report."""
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    pixel_counts = torch.tensor(
        [audit["pixel_counts"][name] for name in class_names],
        dtype=torch.float32,
        device=device,
    )
    weights = calculate_log_class_weights(pixel_counts)
    report = {name: weight.item() for name, weight in zip(class_names, weights, strict=True)}
    return weights, report


def save_training_figure(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save pastel loss and validation Dice curves."""
    epochs = [record["epoch"] for record in history]
    training_loss = [record["training_loss"] for record in history]
    validation_loss = [record["validation_loss"] for record in history]
    semantic_dice = [record["validation_semantic_dice"] for record in history]
    spatial_dice = [record["validation_spatial_dice"] for record in history]
    selection_scores = [record["selection_score"] for record in history]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    figure.patch.set_facecolor("#fff7fb")
    for axis in axes:
        axis.set_facecolor("#fff7fb")
        axis.grid(color="#e8d7e0", alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].plot(
        epochs,
        training_loss,
        color="#d98ba6",
        marker="o",
        label="Training total",
    )
    axes[0].plot(
        epochs,
        validation_loss,
        color="#b7a0d8",
        marker="o",
        label="Validation total",
    )
    axes[0].set_title("Multitask loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        epochs,
        semantic_dice,
        color="#d98ba6",
        marker="o",
        label="Semantic foreground",
    )
    axes[1].plot(
        epochs,
        spatial_dice,
        color="#8fb8d8",
        marker="o",
        label="Spatial foreground",
    )
    axes[1].plot(
        epochs,
        selection_scores,
        color="#a8c7a5",
        marker="o",
        label="Selection score",
    )
    axes[1].set_title("Fold 2 validation Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro Dice")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    figure.suptitle("PanNuke boundary-aware U-Net training")
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def main() -> None:
    """Train the model and save its best validation checkpoint."""
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    set_random_seeds()
    device = torch.device("cuda")
    epochs = choose_setting(arguments.epochs, 2, 25, arguments.smoke_test)
    batch_size = choose_setting(arguments.batch_size, 2, 4, arguments.smoke_test)
    base_channels = choose_setting(
        arguments.base_channels,
        16,
        32,
        arguments.smoke_test,
    )
    train_limit = (
        arguments.train_limit
        if arguments.train_limit is not None
        else (32 if arguments.smoke_test else None)
    )
    validation_limit = (
        arguments.validation_limit
        if arguments.validation_limit is not None
        else (16 if arguments.smoke_test else None)
    )
    default_name = "boundary_unet_smoke" if arguments.smoke_test else "boundary_unet"
    run_name = arguments.run_name or default_name

    report_path = Path(f"reports/{run_name}_training.json")
    figure_path = Path(f"reports/figures/{run_name}_training_curve.png")
    checkpoint_path = Path(f"models/checkpoints/{run_name}_best.pt")
    last_checkpoint_path = Path(f"models/checkpoints/{run_name}_last.pt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    training_loader, validation_loader = create_data_loaders(
        batch_size,
        train_limit,
        validation_limit,
    )
    model = BoundaryAwareUNet(
        input_channels=3,
        number_of_semantic_classes=6,
        number_of_spatial_classes=3,
        base_channels=base_channels,
    ).to(device)

    semantic_weights, semantic_weight_report = load_class_weights(
        SEMANTIC_AUDIT_PATH,
        DEFAULT_CLASS_NAMES,
        device,
    )
    spatial_weights, spatial_weight_report = load_class_weights(
        SPATIAL_AUDIT_PATH,
        SPATIAL_CLASS_NAMES,
        device,
    )
    print("Semantic class weights:")
    for name, weight in semantic_weight_report.items():
        print(f"  {name}: {weight:.4f}")
    print("Spatial class weights:")
    for name, weight in spatial_weight_report.items():
        print(f"  {name}: {weight:.4f}")

    criterion = BoundaryAwareSegmentationLoss(
        semantic_class_weights=semantic_weights,
        spatial_class_weights=spatial_weights,
        semantic_task_weight=SEMANTIC_TASK_WEIGHT,
        spatial_task_weight=SPATIAL_TASK_WEIGHT,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        weight_decay=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    scaler = torch.amp.GradScaler("cuda")

    history: list[dict[str, Any]] = []
    best_selection_score = -1.0
    start_epoch = 1

    if arguments.resume:
        if not last_checkpoint_path.exists():
            raise FileNotFoundError(f"No resumable checkpoint found at {last_checkpoint_path}.")
        checkpoint = torch.load(
            last_checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint["history"]
        best_selection_score = checkpoint["best_selection_score"]
        start_epoch = checkpoint["epoch"] + 1

        generator_state = checkpoint.get("data_generator_state")
        if generator_state is not None and training_loader.generator is not None:
            training_loader.generator.set_state(generator_state.cpu())
        print(f"Resuming after epoch {checkpoint['epoch']}.")

    print("GPU:", torch.cuda.get_device_name(device))
    print("Training samples:", len(training_loader.dataset))
    print("Validation samples:", len(validation_loader.dataset))
    print("Epochs:", epochs)
    print("Batch size:", batch_size)
    print("Semantic task weight:", SEMANTIC_TASK_WEIGHT)
    print("Spatial task weight:", SPATIAL_TASK_WEIGHT)
    print("------------------------")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        training = train_boundary_epoch(
            model=model,
            data_loader=training_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )
        validation = evaluate_boundary_model(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )
        duration_seconds = time.perf_counter() - epoch_start
        scheduler.step(validation["loss"])

        semantic_dice = validation["semantic"]["macro_foreground_dice"] or 0.0
        spatial_dice = validation["spatial"]["macro_foreground_dice"] or 0.0
        selection_score = validation["selection_score"]
        record = {
            "epoch": epoch,
            "training_loss": training["loss"],
            "training_semantic_loss": training["semantic_loss"],
            "training_spatial_loss": training["spatial_loss"],
            "validation_loss": validation["loss"],
            "validation_semantic_loss": validation["semantic_loss"],
            "validation_spatial_loss": validation["spatial_loss"],
            "validation_semantic_dice": semantic_dice,
            "validation_spatial_dice": spatial_dice,
            "selection_score": selection_score,
            "semantic_metrics": validation["semantic"],
            "spatial_metrics": validation["spatial"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "duration_seconds": duration_seconds,
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train={training['loss']:.4f} | "
            f"validation={validation['loss']:.4f} | "
            f"semantic Dice={semantic_dice:.4f} | "
            f"spatial Dice={spatial_dice:.4f} | "
            f"score={selection_score:.4f} | "
            f"time={duration_seconds / 60.0:.1f} min"
        )

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation,
                    "selection_score": selection_score,
                    "dataset_revision": REVISION,
                },
                checkpoint_path,
            )

        last_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "history": history,
            "best_selection_score": best_selection_score,
            "dataset_revision": REVISION,
            "data_generator_state": (
                training_loader.generator.get_state()
                if training_loader.generator is not None
                else None
            ),
        }
        torch.save(last_checkpoint, last_checkpoint_path)

        report = {
            "run_name": run_name,
            "purpose": (
                "boundary-aware pipeline smoke test"
                if arguments.smoke_test
                else "boundary-aware U-Net training"
            ),
            "dataset": DATASET_NAME,
            "revision": REVISION,
            "training_split": "fold1",
            "validation_split": "fold2",
            "fold3_used": False,
            "seed": SEED,
            "epochs": epochs,
            "batch_size": batch_size,
            "base_channels": base_channels,
            "training_samples": len(training_loader.dataset),
            "validation_samples": len(validation_loader.dataset),
            "semantic_class_weights": semantic_weight_report,
            "spatial_class_weights": spatial_weight_report,
            "semantic_task_weight": SEMANTIC_TASK_WEIGHT,
            "spatial_task_weight": SPATIAL_TASK_WEIGHT,
            "checkpoint_selection": ("mean of semantic and spatial macro foreground Dice"),
            "best_selection_score": best_selection_score,
            "history": history,
            "subset_sampling": "seeded random without replacement",
        }
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        save_training_figure(history, figure_path)

    print("------------------------")
    print("Best validation selection score:", round(best_selection_score, 4))
    print("Saved checkpoint:", checkpoint_path)
    print("Saved report:", report_path)
    print("Saved figure:", figure_path)
    print("Saved resumable checkpoint:", last_checkpoint_path)
    print("Training run completed!")


if __name__ == "__main__":
    main()
