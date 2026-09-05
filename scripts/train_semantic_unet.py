"""Train and validate the semantic PanNuke U-Net."""

import argparse
import json
import random
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
from pancancer_nuclei.models.losses import (
    CombinedSegmentationLoss,
    calculate_log_class_weights,
)
from pancancer_nuclei.models.unet import UNet
from pancancer_nuclei.training.engine import (
    evaluate_semantic_model,
    segmentation_collate,
    train_one_epoch,
)

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = "data/raw/huggingface_cache"
CLASS_AUDIT_PATH = Path("reports/fold1_semantic_pixel_audit.json")
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
    parser.add_argument("--class-balanced", action="store_true")
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

    training_dataset = limit_dataset(
        training_dataset,
        train_limit,
        seed=SEED,
    )
    validation_dataset = limit_dataset(
        validation_dataset,
        validation_limit,
        seed=SEED + 1,
    )

    generator = torch.Generator().manual_seed(SEED)

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=segmentation_collate,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=segmentation_collate,
    )

    return training_loader, validation_loader


def save_training_figure(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save pastel training and validation curves."""
    epochs = [record["epoch"] for record in history]
    training_losses = [record["training_loss"] for record in history]
    validation_losses = [record["validation_loss"] for record in history]
    validation_dice = [record["validation_macro_foreground_dice"] for record in history]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    figure.patch.set_facecolor("#fff7fb")

    for axis in axes:
        axis.set_facecolor("#fff7fb")
        axis.grid(color="#e8d7e0", alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].plot(
        epochs,
        training_losses,
        color="#d98ba6",
        marker="o",
        label="Training",
    )
    axes[0].plot(
        epochs,
        validation_losses,
        color="#b7a0d8",
        marker="o",
        label="Validation",
    )
    axes[0].set_title("Segmentation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Combined loss")
    axes[0].legend()

    axes[1].plot(
        epochs,
        validation_dice,
        color="#a8c7a5",
        marker="o",
    )
    axes[1].set_title("Validation foreground Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro Dice")
    axes[1].set_ylim(0, 1)

    figure.suptitle("PanNuke semantic U-Net training")
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

    epochs = choose_setting(
        arguments.epochs,
        smoke_value=2,
        full_value=25,
        smoke_test=arguments.smoke_test,
    )
    batch_size = choose_setting(
        arguments.batch_size,
        smoke_value=2,
        full_value=4,
        smoke_test=arguments.smoke_test,
    )
    base_channels = choose_setting(
        arguments.base_channels,
        smoke_value=16,
        full_value=32,
        smoke_test=arguments.smoke_test,
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

    default_run_name = "unet_smoke" if arguments.smoke_test else "semantic_unet"
    run_name = arguments.run_name or default_run_name

    report_path = Path(f"reports/{run_name}_training.json")
    figure_path = Path(f"reports/figures/{run_name}_training_curve.png")
    checkpoint_path = Path(f"models/checkpoints/{run_name}_best.pt")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    training_loader, validation_loader = create_data_loaders(
        batch_size=batch_size,
        train_limit=train_limit,
        validation_limit=validation_limit,
    )

    model = UNet(
        input_channels=3,
        number_of_classes=6,
        base_channels=base_channels,
    ).to(device)
    class_weights = None
    class_weight_report = None

    if arguments.class_balanced:
        audit = json.loads(CLASS_AUDIT_PATH.read_text(encoding="utf-8"))
        pixel_counts = torch.tensor(
            [audit["pixel_counts"][class_name] for class_name in DEFAULT_CLASS_NAMES],
            dtype=torch.float32,
            device=device,
        )
        class_weights = calculate_log_class_weights(pixel_counts)
        class_weight_report = {
            class_name: weight.item()
            for class_name, weight in zip(
                DEFAULT_CLASS_NAMES,
                class_weights,
                strict=True,
            )
        }

        print("Class weights:")
        for class_name, weight in class_weight_report.items():
            print(f"  {class_name}: {weight:.4f}")

    criterion = CombinedSegmentationLoss(
        class_weights=class_weights,
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
    best_validation_dice = -1.0

    print("GPU:", torch.cuda.get_device_name(device))
    print("Training samples:", len(training_loader.dataset))
    print("Validation samples:", len(validation_loader.dataset))
    print("Epochs:", epochs)
    print("Batch size:", batch_size)
    print("------------------------")

    for epoch in range(1, epochs + 1):
        training_loss = train_one_epoch(
            model=model,
            data_loader=training_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )
        validation_metrics = evaluate_semantic_model(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        validation_loss = validation_metrics["loss"]
        validation_dice = validation_metrics["macro_foreground_dice"] or 0.0
        scheduler.step(validation_loss)

        record = {
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "validation_macro_foreground_dice": validation_dice,
            "validation_confusion_matrix": validation_metrics["confusion_matrix"],
            "validation_pixel_accuracy": validation_metrics["pixel_accuracy"],
            "validation_dice_per_class": validation_metrics["dice_per_class"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss={training_loss:.4f} | "
            f"validation loss={validation_loss:.4f} | "
            f"validation Dice={validation_dice:.4f}"
        )

        if validation_dice > best_validation_dice:
            best_validation_dice = validation_dice
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "dataset_revision": REVISION,
                },
                checkpoint_path,
            )

        report = {
            "run_name": run_name,
            "purpose": (
                "pipeline smoke test"
                if arguments.smoke_test
                else "semantic U-Net baseline training"
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
            "best_validation_macro_foreground_dice": (best_validation_dice),
            "history": history,
            "subset_sampling": "seeded random without replacement",
            "class_balanced": arguments.class_balanced,
            "class_weights": class_weight_report,
        }
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        save_training_figure(history, figure_path)

    print("------------------------")
    print(
        "Best validation foreground Dice:",
        round(best_validation_dice, 4),
    )
    print("Saved checkpoint:", checkpoint_path)
    print("Saved report:", report_path)
    print("Saved figure:", figure_path)
    print("Training run completed!")


if __name__ == "__main__":
    main()
