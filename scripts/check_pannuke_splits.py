from datasets import load_dataset

DATASET_NAME = "RationAI/PanNuke"
REVISION = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
SPLITS = ("fold1", "fold2", "fold3")

for split in SPLITS:
    print(f"\nChecking {split}...")

    dataset = load_dataset(
        DATASET_NAME,
        split=split,
        revision=REVISION,
        streaming=True,
    )

    sample = next(iter(dataset))

    tissue_value = sample["tissue"]
    if isinstance(tissue_value, int):
        tissue_name = dataset.features["tissue"].int2str(tissue_value)
    else:
        tissue_name = str(tissue_value)

    print("Fields:", list(sample.keys()))
    print("Tissue:", tissue_name)
    print("Number of nuclei:", len(sample["instances"]))
    print("Number of labels:", len(sample["categories"]))

    assert len(sample["instances"]) == len(sample["categories"])

print("\nAll three PanNuke folds are accessible.")
