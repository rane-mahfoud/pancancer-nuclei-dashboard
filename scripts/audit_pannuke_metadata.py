import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
CACHE_DIR = Path("data/raw/huggingface_cache")
OUTPUT_PATH = Path("reports/data_audit_metadata.json")

CLASS_NAMES = {
    0: "neoplastic",
    1: "inflammatory",
    2: "connective",
    3: "dead",
    4: "epithelial",
}

dataset = load_dataset(
    DATASET_NAME,
    revision=REVISION,
    cache_dir=str(CACHE_DIR),
)

audit = {
    "dataset": DATASET_NAME,
    "revision": REVISION,
    "total_samples": 0,
    "total_nuclei": 0,
    "empty_samples": 0,
    "class_counts": {},
    "tissue_counts": {},
    "splits": {},
}

all_class_counts = Counter()
all_tissue_counts = Counter()

for split_name in ("fold1", "fold2", "fold3"):
    split = dataset[split_name]
    metadata = split.select_columns(["categories", "tissue"])
    tissue_feature = split.features["tissue"]

    class_counts = Counter()
    tissue_counts = Counter()
    empty_samples = 0
    invalid_categories = Counter()

    for row in tqdm(metadata, desc=f"Auditing {split_name}"):
        categories = row["categories"]

        if not categories:
            empty_samples += 1

        for category in categories:
            if category in CLASS_NAMES:
                class_counts[category] += 1
            else:
                invalid_categories[category] += 1

        tissue_value = row["tissue"]
        if isinstance(tissue_value, int):
            tissue_name = tissue_feature.int2str(tissue_value)
        else:
            tissue_name = str(tissue_value)

        tissue_counts[tissue_name] += 1

    assert not invalid_categories
    assert sum(class_counts.values()) == sum(
        len(categories) for categories in metadata["categories"]
    )

    named_class_counts = {CLASS_NAMES[class_id]: class_counts[class_id] for class_id in CLASS_NAMES}

    audit["splits"][split_name] = {
        "samples": len(split),
        "nuclei": sum(class_counts.values()),
        "empty_samples": empty_samples,
        "class_counts": named_class_counts,
        "tissue_counts": dict(sorted(tissue_counts.items())),
    }

    audit["total_samples"] += len(split)
    audit["total_nuclei"] += sum(class_counts.values())
    audit["empty_samples"] += empty_samples

    all_class_counts.update(class_counts)
    all_tissue_counts.update(tissue_counts)

audit["class_counts"] = {
    CLASS_NAMES[class_id]: all_class_counts[class_id] for class_id in CLASS_NAMES
}
audit["tissue_counts"] = dict(sorted(all_tissue_counts.items()))

assert audit["total_samples"] == 7901
assert sum(audit["class_counts"].values()) == audit["total_nuclei"]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(
    json.dumps(audit, indent=2),
    encoding="utf-8",
)

print("\nMetadata audit completed")
print("------------------------")
print("Total samples:", audit["total_samples"])
print("Total nuclei:", audit["total_nuclei"])
print("Empty samples:", audit["empty_samples"])
print("Class counts:", audit["class_counts"])
print("Tissue types:", len(audit["tissue_counts"]))
print("Saved to:", OUTPUT_PATH)
