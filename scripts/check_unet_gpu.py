"""Run one real PanNuke training step on the GPU."""

import torch
from datasets import load_dataset

from pancancer_nuclei.data.pannuke import IGNORE_LABEL, PanNukeDataset
from pancancer_nuclei.data.transforms import create_training_transforms
from pancancer_nuclei.models.losses import CombinedSegmentationLoss
from pancancer_nuclei.models.unet import UNet

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = "data/raw/huggingface_cache"


def main() -> None:
    """Run a forward and backward pass using one real image."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    device = torch.device("cuda")
    torch.manual_seed(42)

    records = load_dataset(
        DATASET_NAME,
        revision=REVISION,
        split="fold1",
        cache_dir=CACHE_DIR,
    )
    dataset = PanNukeDataset(
        records,
        transform=create_training_transforms(),
    )
    sample = dataset[0]

    image = sample["image"].unsqueeze(0).to(device)
    target = sample["semantic_mask"].unsqueeze(0).to(device)

    model = UNet(
        input_channels=3,
        number_of_classes=6,
        base_channels=32,
    ).to(device)
    model.train()

    torch.cuda.reset_peak_memory_stats(device)

    predictions = model(image)

    criterion = CombinedSegmentationLoss(
        ignore_label=IGNORE_LABEL,
    )
    loss = criterion(predictions, target)

    loss.backward()
    torch.cuda.synchronize()

    gradient = model.classifier.weight.grad

    assert predictions.shape == (1, 6, 256, 256)
    assert torch.isfinite(loss)
    assert gradient is not None
    assert torch.isfinite(gradient).all()

    parameters = sum(parameter.numel() for parameter in model.parameters())
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1024**3

    print("------------------------")
    print("GPU:", torch.cuda.get_device_name(device))
    print("Tissue:", sample["tissue"])
    print("Input shape:", tuple(image.shape))
    print("Output shape:", tuple(predictions.shape))
    print("Trainable parameters:", f"{parameters:,}")
    print("Loss:", round(loss.item(), 4))
    print("Peak GPU memory GB:", round(peak_memory_gb, 3))
    print("Forward and backward GPU check passed!")


if __name__ == "__main__":
    main()
