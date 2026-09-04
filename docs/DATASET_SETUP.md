# PanNuke dataset setup

## Why PanNuke is the MVP dataset

PanNuke supplies 256 × 256 H&E patches, instance-level masks for five nucleus categories, tissue labels spanning 19 tissue types, and three predefined folds. This supports a compact but credible study of instance segmentation, phenotype classification, cross-tissue reliability, and downstream counting.

Do not add CoNIC during the MVP. Its six colon-specific classes and challenge protocol create a second label ontology and would turn a ten-day project into a harmonization project.

## Sources

- Dataset paper and schema: https://arxiv.org/abs/2003.10778
- University of Warwick legacy dataset page: https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke/
- Official archived metric implementation: https://github.com/TissueImageAnalytics/PanNuke-metrics
- Practical public mirror, if the legacy host is unavailable: https://huggingface.co/datasets/RationAI/PanNuke

The mirror is a convenience distribution, not a replacement for reading/citing the original paper. Record the mirror revision or file hashes used.

## License and repository policy

The dataset card identifies PanNuke as **CC BY-NC-SA 4.0**. Before downloading, read the dataset terms yourself.

- Never commit raw images, masks, archives, or model checkpoints trained on the data.
- Keep the code license and dataset license conceptually separate.
- Attribute PanNuke in the README, report, and any dataset-derived figure.
- If sample images or overlays are later committed, label them as PanNuke-derived assets and preserve compatible attribution/licensing.
- Do not present the data or predictions as suitable for clinical use.

## Expected local layout

Place original-format files outside version control:

```text
data/raw/pannuke/
├── fold1/
│   ├── images.npy
│   ├── masks.npy
│   └── types.npy
├── fold2/
│   ├── images.npy
│   ├── masks.npy
│   └── types.npy
└── fold3/
    ├── images.npy
    ├── masks.npy
    └── types.npy
```

If the downloaded archive uses a different nesting pattern, write one conversion script. Do not manually rearrange thousands of samples.

Expected original mask representation:

- images: `N × 256 × 256 × 3`;
- masks: `N × 256 × 256 × 6`;
- positive mask channels 0–4 contain integer instance IDs;
- channel order: neoplastic, inflammatory, connective, dead, non-neoplastic epithelial;
- the sixth channel represents background in common original distributions;
- tissue types align one-to-one with image rows.

Treat these as hypotheses to verify in the audit, not assumptions to force onto unknown files.

## Download route A — original fold arrays

1. Open the Warwick/TIA source linked above.
2. Read the terms and download all three folds.
3. Place each fold under the expected layout.
4. Preserve the original archives until checksums and conversion are complete.
5. Record filenames, sizes, and SHA-256 hashes in `data/manifest.local.json`.
6. Keep the manifest local if source filenames contain private tokens; otherwise commit only hashes and public filenames.

## Download route B — Hugging Face mirror

Use this only if the official download is inaccessible. Install the optional `datasets` dependency, load the three named folds, and convert every example to the repository’s canonical image/instance/type representation. Pin a dataset revision in the conversion configuration.

The following command pins the mirror state that was current when this plan was prepared:

```bash
python -m pip install -e ".[data-mirror]"
python - <<'PY'
from datasets import load_dataset

revision = "1f498f7bd6a85ef5f204c592b41ac881eab61005"
dataset = load_dataset("RationAI/PanNuke", revision=revision)

print(dataset)
print(dataset["fold1"].features)
for split in ("fold1", "fold2", "fold3"):
    print(split, len(dataset[split]))
PY
```

The mirror schema documented at that revision is:

- `image`: one 256 × 256 RGB patch;
- `instances`: a list of binary masks, one per nucleus;
- `categories`: one integer per instance, ordered `0` neoplastic, `1` inflammatory, `2` connective, `3` dead, `4` epithelial;
- `tissue`: one of 19 tissue labels.

Verify this schema from `dataset["fold1"].features` and sample-level assertions. Do not treat the prose card as a substitute for runtime validation.

Because the mirror exposes instances and category lists rather than necessarily reproducing the original six-channel arrays byte-for-byte, the conversion must be tested on multiple examples. Never mix original and mirror samples within a reported run.

## Mandatory audit before modeling

Produce `reports/data_audit.json` and `reports/figures/data_audit_grid.png` with the following checks:

- [ ] three folds found and non-empty;
- [ ] image/mask/type row counts match within each fold;
- [ ] image height, width, and channel count are correct;
- [ ] image dtype and intensity range recorded;
- [ ] mask dtype, channel count, and min/max recorded;
- [ ] tissue label vocabulary has 19 expected categories or discrepancies are explained;
- [ ] per-class instance counts computed from unique nonzero IDs;
- [ ] per-tissue patch counts computed;
- [ ] duplicate hashes within and across folds reported;
- [ ] positive-class pixel overlap counted;
- [ ] empty-mask patches counted;
- [ ] every instance has one non-empty connected region, or violations quantified;
- [ ] at least 24 image/overlay pairs inspected across tissues and classes;
- [ ] background reconstruction verified rather than trusted blindly.

## Split contract

For the first completed experiment:

- Fold 1 = training;
- Fold 2 = validation and post-processing calibration;
- Fold 3 = locked test.

Do not inspect Fold 3 performance during development. It is acceptable to validate file readability and shapes, but not to use test results to choose a model, threshold, figure style, or “best” epoch.

## Data loader tests

The test suite must establish that:

1. an image and every target remain spatially aligned after each augmentation;
2. rotations and flips preserve discrete instance IDs;
3. semantic class IDs stay in `[0, 5]`;
4. globally relabeled instance IDs are consecutive and do not merge classes;
5. generated boundary pixels lie inside or directly on annotated instances according to the declared convention;
6. no normalization is applied twice;
7. a fixed seed returns the same validation sample.

## Data-card notes for the final README

Report dataset provenance, magnification/resolution statements from the paper, annotation process, label schema, split selection, license, and known limits. Do not imply that patches are independent patients or that tissue-wise evaluation is site-wise external validation.
