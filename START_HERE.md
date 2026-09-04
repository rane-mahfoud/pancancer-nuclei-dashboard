# Start here

This is the planning and repository scaffold for the first portfolio project:
**Pan-cancer nucleus instance segmentation, phenotyping, and cell-composition profiling**.

## The one-sentence project

> I test whether a lightweight boundary-aware U-Net can recover nucleus instances and phenotypes across heterogeneous PanNuke tissues, and whether its segmentation errors distort downstream cell-composition measurements.

## The project is successful when

1. A new user can reproduce a smoke test from a fresh environment.
2. The data audit proves that labels, tissue types, and official folds are handled correctly.
3. A semantic-only U-Net and a boundary-aware U-Net are compared under the same split, encoder, augmentations, and training budget.
4. The untouched test fold is evaluated once with bPQ and mPQ as primary metrics.
5. Results are broken down by tissue, nucleus class, and cell-density regime.
6. Count errors and at least one morphometric feature are reported, not just pixel overlap.
7. The dashboard runs from small, precomputed held-out examples and exports a transparent CSV.
8. The README contains real results, representative successes, worst failures, limitations, and exact reproduction commands.

## Work in this order

1. Read `docs/RESEARCH_PROTOCOL.md`.
2. Complete Day 1 in `docs/DAILY_CHECKLIST.md`.
3. Copy the issues from `docs/GITHUB_ISSUES.md` into GitHub.
4. Do not train a full model until the dataset audit and target tests pass.
5. Do not inspect the final test metrics until preprocessing, model choice, and post-processing thresholds are frozen.

## Scope guardrails

- **Dataset:** PanNuke only for the MVP. CoNIC is a later external-validation extension.
- **Core comparison:** semantic-only U-Net vs the same U-Net with an additional boundary output.
- **Primary metrics:** binary panoptic quality (bPQ) and multiclass panoptic quality (mPQ).
- **Secondary metrics:** class-wise PQ, detection F1, foreground Dice, count MAE/bias, and runtime.
- **No clinical claim:** cell-composition outputs are quantitative image-derived features, not validated biomarkers or diagnoses.
- **No fake completeness:** every unavailable result remains `TBD`; no placeholder number may survive the public release.

## Immediate first task

Create the repository, commit this scaffold, create the environment, and download only enough data to validate one sample. The first meaningful milestone is not “model trained”; it is “one image passes through a tested image → target → prediction-format → metric pipeline.”
