import albumentations as A
import numpy as np
import pytest
import torch

from pancancer_nuclei.data.pannuke import (
    IGNORE_LABEL,
    PanNukeDataset,
    prepare_pannuke_sample,
)


def test_prepare_pannuke_sample() -> None:
    image = np.full((4, 4, 3), 255, dtype=np.uint8)

    first_mask = np.zeros((4, 4), dtype=bool)
    first_mask[0, 0] = True
    first_mask[1, 1] = True

    second_mask = np.zeros((4, 4), dtype=bool)
    second_mask[1, 1] = True
    second_mask[2, 2] = True

    sample = {
        "image": image,
        "instances": [first_mask, second_mask],
        "categories": [0, 2],
        "tissue": "Breast",
    }

    result = prepare_pannuke_sample(sample)

    assert result["image"].shape == (3, 4, 4)
    assert result["image"].dtype == torch.float32
    assert result["image"].min().item() == 1.0
    assert result["image"].max().item() == 1.0

    assert result["semantic_mask"][0, 0].item() == 1
    assert result["semantic_mask"][2, 2].item() == 3
    assert result["semantic_mask"][3, 3].item() == 0
    assert result["semantic_mask"][1, 1].item() == IGNORE_LABEL

    assert result["instance_map"][0, 0].item() == 1
    assert result["instance_map"][2, 2].item() == 2
    assert result["instance_map"][1, 1].item() == 0

    assert result["instance_masks"].shape == (2, 4, 4)
    assert result["categories"].tolist() == [0, 2]
    assert result["tissue"] == "Breast"


def test_prepare_empty_sample() -> None:
    sample = {
        "image": np.zeros((4, 4, 3), dtype=np.uint8),
        "instances": [],
        "categories": [],
        "tissue": "Colon",
    }

    result = prepare_pannuke_sample(sample)

    assert result["instance_masks"].shape == (0, 4, 4)
    assert result["semantic_mask"].sum().item() == 0
    assert result["instance_map"].sum().item() == 0
    assert not result["overlap_mask"].any()


def test_rejects_mismatched_annotations() -> None:
    sample = {
        "image": np.zeros((4, 4, 3), dtype=np.uint8),
        "instances": [np.zeros((4, 4), dtype=bool)],
        "categories": [],
        "tissue": "Breast",
    }

    with pytest.raises(ValueError, match="number of instance masks"):
        prepare_pannuke_sample(sample)


def test_transformation_keeps_image_and_mask_aligned() -> None:
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[1, 0] = [255, 0, 0]

    nucleus_mask = np.zeros((3, 4), dtype=bool)
    nucleus_mask[1, 0] = True

    records = [
        {
            "image": image,
            "instances": [nucleus_mask],
            "categories": [0],
            "tissue": "Breast",
        }
    ]

    horizontal_flip = A.Compose([A.HorizontalFlip(p=1.0)])
    dataset = PanNukeDataset(records, transform=horizontal_flip)

    result = dataset[0]

    assert result["image"][0, 1, 3].item() == 1.0
    assert result["instance_masks"][0, 1, 3].item()
    assert result["semantic_mask"][1, 3].item() == 1

    assert result["image"][0, 1, 0].item() == 0.0
    assert not result["instance_masks"][0, 1, 0].item()
