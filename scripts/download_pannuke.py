from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = Path("data/raw/huggingface_cache")

CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading PanNuke to: {CACHE_DIR.resolve()}")

dataset = load_dataset(
    DATASET_NAME,
    revision=REVISION,
    cache_dir=str(CACHE_DIR),
)

print("\nDownloaded dataset")
print("------------------")

for split_name, split_data in dataset.items():
    print(f"{split_name}: {len(split_data):,} samples")

print("\nTotal samples:", sum(len(split) for split in dataset.values()))
print("Download and split checks completed.")
