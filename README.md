# PanCancer Nuclei Profiling

**Lightweight nucleus instance segmentation, phenotyping, and cell-composition analysis across heterogeneous H&E tissues**

> Status: repository scaffold and preregistered analysis plan. Numerical results are intentionally marked `TBD` until experiments are run.

## Why this project

Nucleus-level analysis can convert histology images into interpretable measurements such as cell counts, phenotype proportions, density, and morphology. A model can nevertheless obtain a reasonable aggregate score while failing on rare cell classes, crowded regions, or particular tissues. Those failures may directly bias the biological measurements produced downstream.

This project asks:

> How reliably can a lightweight boundary-aware U-Net recover nucleus instances and phenotypes across PanNuke tissue types, and how do segmentation and classification errors propagate into cell-composition estimates?

The goal is a transparent, reproducible study of a practical model—not a claim of state-of-the-art performance or clinical readiness.

## Study design

Two controlled systems are compared:

| System | Network output | Instance reconstruction | Purpose |
| --- | --- | --- | --- |
| E1: semantic baseline | Six-class semantic logits | Connected components | Establish the failure caused by touching nuclei |
| E2: boundary-aware model | Six-class semantic logits + boundary logit | Marker-controlled watershed | Test whether explicit separation improves instance and counting quality |

Both systems use the same encoder, data split, augmentations, optimizer, and training budget. Post-processing thresholds are selected on validation data and frozen before final test evaluation.

## Dataset

The study uses **PanNuke**, a pan-cancer H&E dataset with nucleus instance masks, five positive nucleus categories, and 19 tissue types. The positive-channel order in the original metric implementation is:

1. neoplastic
2. inflammatory
3. connective/soft tissue
4. dead
5. non-neoplastic epithelial

Images and raw masks are never committed to this repository. See [`docs/DATASET_SETUP.md`](docs/DATASET_SETUP.md) for access, expected layout, validation checks, and licensing notes.

## Evaluation

The primary endpoints are:

- **bPQ:** panoptic quality when all nuclei are treated as one class;
- **mPQ:** per-class PQ averaged across positive nucleus classes, with tissue-macro reporting.

Secondary analyses include class-wise PQ, detection precision/recall/F1, foreground Dice, per-class count MAE and bias, density-stratified performance, inference time, and qualitative split/merge/misclassification failures.

The final test set is evaluated only after the analysis contract is frozen. Full definitions are in [`docs/EVALUATION_PLAN.md`](docs/EVALUATION_PLAN.md).

## Planned results

### Main comparison

| Experiment | Validation bPQ | Test bPQ | Test mPQ | Detection F1 | Mean count MAE | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| E1 semantic + connected components | TBD | TBD | TBD | TBD | TBD | Not run |
| E2 semantic + boundary + watershed | TBD | TBD | TBD | TBD | TBD | Not run |

### Required stratified outputs

- PQ by nucleus class;
- bPQ and mPQ by tissue type;
- count error by class and tissue;
- performance by ground-truth nucleus-density quartile;
- the five best and ten worst held-out patches with a failure label.

No result will be added without the run configuration, seed, checkpoint identifier, and machine/GPU information.

## Dashboard

The Streamlit app will operate on a small set of precomputed held-out examples so that it remains reproducible without redistributing the full dataset or a large checkpoint. It will provide:

- image, reference, and prediction overlays;
- per-class cell counts and proportions;
- nucleus area/equivalent-diameter summaries in pixels;
- confidence and limitations panel;
- downloadable per-instance CSV.

The dashboard is a research demonstration. It does not provide a diagnosis or a clinically validated biomarker.

## Repository structure

```text
.
├── app/                       # Streamlit application
├── configs/                   # Versioned experiment configurations
├── data/                      # Ignored raw/interim/processed data locations
├── docs/                      # Protocol, setup, issues, evaluation, schedule
├── models/checkpoints/        # Ignored model weights
├── notebooks/                 # Exploration only; no production logic
├── reports/figures/           # Generated, publication-ready figures
├── scripts/                   # Thin command-line entry points
├── src/pancancer_nuclei/      # Reusable package code
└── tests/                     # Unit and integration tests
```

## Quick start

```bash
conda env create -f environment.yml
conda activate pancancer-nuclei
python -m pip install -e .
pytest -q
```

Dataset preparation and training commands will be activated as their corresponding issues are completed. Until then, follow [`START_HERE.md`](START_HERE.md) and the [`daily checklist`](docs/DAILY_CHECKLIST.md).

## Reproducibility contract

- Random seeds and deterministic settings are logged.
- Raw data stay outside Git and are addressed through a configurable path.
- Every run writes its resolved configuration, environment metadata, metrics, and checkpoint hash.
- Test data are not used for threshold selection, early stopping, or figure-driven model changes.
- A one-batch overfit test and a CPU smoke test must pass before full training.
- Exact package locks are generated only after the environment succeeds on the actual training platform.

## Limitations known in advance

- PanNuke labels are semi-automatically generated and quality-controlled, not an error-free ground truth.
- Patch-level evaluation does not establish whole-slide or patient-level clinical utility.
- Tissue types are heterogeneous, while patient/site identifiers needed for some stronger leakage analyses may not be available.
- Rare classes—especially dead cells—make aggregate accuracy and pixel metrics misleading.
- Pixel-derived morphology is not a physical measurement unless scale metadata are available and validated.
- A cell-composition profile is not a validated diagnostic or prognostic biomarker.

## Data, code, and attribution

PanNuke is distributed separately under **CC BY-NC-SA 4.0**. The original data are not included here. Code in this repository is planned for release under the MIT License; dataset-derived visual assets must retain PanNuke attribution and compatible licensing.

Key references:

- Gamper J. et al. *PanNuke: An Open Pan-Cancer Histology Dataset for Nuclei Instance Segmentation and Classification* (2019).
- Gamper J. et al. *PanNuke Dataset Extension, Insights and Baselines* (2020), arXiv:2003.10778.
- Graham S. et al. *HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images* (2019).

## Author

Rane Mahfoud — biomedical imaging researcher with an MSc in Biotechnical Medical Systems and Technologies. Prior work includes breast-cancer cell-image segmentation with U-Net, StarDist, and Cellpose.
