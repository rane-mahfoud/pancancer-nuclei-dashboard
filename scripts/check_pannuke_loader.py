"""Smoke-test the PanNuke PyTorch data conversion using a real sample."""

from datasets import load_dataset

from pancancer_nuclei.data.pannuke import PanNukeDataset

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = "data/raw/huggingface_cache"


dataset = load_dataset(
    DATASET_NAME,
    revision=REVISION,
    split="fold1",
    cache_dir=CACHE_DIR,
)

sample = PanNukeDataset(dataset)[0]

print("Image tensor:", tuple(sample["image"].shape))
print("Semantic mask:", tuple(sample["semantic_mask"].shape))
print("Instance masks:", tuple(sample["instance_masks"].shape))
print("Categories:", sample["categories"].tolist())
print("Tissue:", sample["tissue"])
print("Overlap pixels:", sample["overlap_mask"].sum().item())

assert sample["image"].shape == (3, 256, 256)
assert sample["semantic_mask"].shape == (256, 256)
assert sample["instance_map"].shape == (256, 256)
assert len(sample["instance_masks"]) == len(sample["categories"])

print("Real PanNuke loader check passed!")
