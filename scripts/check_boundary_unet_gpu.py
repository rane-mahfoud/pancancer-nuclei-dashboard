"""Run a forward and backward boundary-aware U-Net GPU check."""

import numpy as np
import torch
from datasets import load_dataset
from torch import nn

from pancancer_nuclei.data.pannuke import PanNukeDataset
from pancancer_nuclei.data.targets import create_boundary_target
from pancancer_nuclei.data.transforms import create_validation_transforms
from pancancer_nuclei.models.boundary_unet import BoundaryAwareUNet
from pancancer_nuclei.models.losses import CombinedSegmentationLoss

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = "data/raw/huggingface_cache"
IGNORE_LABEL = 255


def main() -> None:
    """Verify both output heads and losses on a real sample."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    records = load_dataset(
        DATASET_NAME,
        revision=REVISION,
        split="fold1",
        cache_dir=CACHE_DIR,
    )
    dataset = PanNukeDataset(
        records,
        transform=create_validation_transforms(),
    )
    sample = dataset[0]

    image = sample["image"].unsqueeze(0).to(device)
    semantic_target = sample["semantic_mask"].unsqueeze(0).to(device)

    instance_masks = sample["instance_masks"].numpy()
    spatial_target_array = create_boundary_target(
        instance_masks,
        boundary_width=2,
    )
    spatial_target = torch.from_numpy(spatial_target_array.astype(np.int64)).unsqueeze(0).to(device)

    model = BoundaryAwareUNet(
        base_channels=32,
    ).to(device)

    semantic_loss_function = CombinedSegmentationLoss()
    spatial_loss_function = nn.CrossEntropyLoss(
        ignore_index=IGNORE_LABEL,
    )

    semantic_logits, spatial_logits = model(image)

    semantic_loss = semantic_loss_function(
        semantic_logits,
        semantic_target,
    )
    spatial_loss = spatial_loss_function(
        spatial_logits,
        spatial_target,
    )
    total_loss = semantic_loss + 0.5 * spatial_loss

    total_loss.backward()

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1024**3

    unique_spatial_labels = sorted(np.unique(spatial_target_array).tolist())

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )
    print("Tissue:", sample["tissue"])
    print("Input shape:", tuple(image.shape))
    print(
        "Semantic output:",
        tuple(semantic_logits.shape),
    )
    print(
        "Spatial output:",
        tuple(spatial_logits.shape),
    )
    print("Spatial target labels:", unique_spatial_labels)
    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )
    print(
        "Semantic loss:",
        f"{semantic_loss.item():.4f}",
    )
    print(
        "Spatial loss:",
        f"{spatial_loss.item():.4f}",
    )
    print(
        "Combined loss:",
        f"{total_loss.item():.4f}",
    )
    print(
        "Peak GPU memory GB:",
        f"{peak_memory_gb:.3f}",
    )
    print("Boundary-aware forward and backward check passed!")


if __name__ == "__main__":
    main()
