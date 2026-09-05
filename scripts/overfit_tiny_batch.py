"""Check that the U-Net can memorize two PanNuke training images."""

import json
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
from datasets import load_dataset

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from pancancer_nuclei.data.pannuke import PanNukeDataset
from pancancer_nuclei.models.losses import CombinedSegmentationLoss
from pancancer_nuclei.models.unet import UNet

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = "data/raw/huggingface_cache"

NUMBER_OF_STEPS = 150
LEARNING_RATE = 1.0e-3
SAMPLE_INDICES = (0, 1)
SEED = 42

REPORT_PATH = Path("reports/tiny_batch_overfit.json")
FIGURE_PATH = Path("reports/figures/tiny_batch_overfit.png")


def set_random_seeds() -> None:
    """Make this diagnostic as repeatable as practical."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def save_loss_curve(losses: list[float]) -> None:
    """Save a pink training-loss figure."""
    figure, axis = plt.subplots(figsize=(8, 5))
    figure.patch.set_facecolor("#fff7fb")
    axis.set_facecolor("#fff7fb")

    steps = np.arange(1, len(losses) + 1)
    axis.plot(
        steps,
        losses,
        color="#d98ba6",
        linewidth=2.5,
    )
    axis.fill_between(
        steps,
        losses,
        color="#f2c9d8",
        alpha=0.45,
    )

    axis.set_title("Tiny-batch U-Net overfitting check")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Combined segmentation loss")
    axis.grid(color="#e8d7e0", alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(
        FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def main() -> None:
    """Train repeatedly on two images and confirm loss decreases."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    set_random_seeds()
    device = torch.device("cuda")

    records = load_dataset(
        DATASET_NAME,
        revision=REVISION,
        split="fold1",
        cache_dir=CACHE_DIR,
    )
    dataset = PanNukeDataset(records)
    samples = [dataset[index] for index in SAMPLE_INDICES]

    images = torch.stack([sample["image"] for sample in samples]).to(device)
    targets = torch.stack([sample["semantic_mask"] for sample in samples]).to(device)

    model = UNet(
        input_channels=3,
        number_of_classes=6,
        base_channels=16,
    ).to(device)

    criterion = CombinedSegmentationLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    model.train()
    losses: list[float] = []

    for step in range(1, NUMBER_OF_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)

        predictions = model(images)
        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        if step == 1 or step % 10 == 0:
            print(f"Step {step:03d}/{NUMBER_OF_STEPS}: loss={loss_value:.4f}")

    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction_percent = 100.0 * (initial_loss - final_loss) / initial_loss

    if final_loss >= initial_loss:
        raise RuntimeError("Loss did not decrease during the tiny-batch test.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "purpose": "pipeline diagnostic, not a model result",
        "dataset": DATASET_NAME,
        "revision": REVISION,
        "split": "fold1",
        "sample_indices": list(SAMPLE_INDICES),
        "steps": NUMBER_OF_STEPS,
        "learning_rate": LEARNING_RATE,
        "base_channels": 16,
        "seed": SEED,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_percent": reduction_percent,
        "losses": losses,
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    save_loss_curve(losses)

    print("------------------------")
    print("Initial loss:", round(initial_loss, 4))
    print("Final loss:", round(final_loss, 4))
    print("Loss reduction:", f"{reduction_percent:.1f}%")
    print("Saved report:", REPORT_PATH)
    print("Saved figure:", FIGURE_PATH)
    print("Tiny-batch overfitting check passed!")


if __name__ == "__main__":
    main()
