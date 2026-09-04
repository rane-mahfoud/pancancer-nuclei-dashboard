# Ten-day execution checklist

Assumption: 5–7 focused hours per day and access to one modest CUDA GPU for full training. If no GPU is available, follow the fallback in `MODEL_BASELINE.md` and preserve the same research pipeline on a smaller declared subset.

Each day ends with a commit and a two-minute lab note: what changed, what evidence passed, what failed, and the exact first task for tomorrow.

## Day 1 — Repository, environment, and one verified sample

**Goal:** one sample can be loaded without ambiguity.

- [ ] Create GitHub account/profile if needed; enable two-factor authentication.
- [ ] Create public repo `pancancer-nuclei-dashboard` and clone it.
- [ ] Commit this scaffold as `chore: initialize research repository`.
- [ ] Create the environment and install the package in editable mode.
- [ ] Run the empty/synthetic test suite and lint check.
- [ ] Read PanNuke license and both PanNuke papers' dataset/evaluation sections.
- [ ] Download one fold or one small mirror slice.
- [ ] Load one image, mask, and tissue label; print shapes, dtypes, and ranges.
- [ ] Render one image with instance boundaries and a class legend.
- [ ] Start a lab note with hardware, Python, PyTorch, and CUDA versions.

**End-of-day evidence:** installation command succeeds; one correctly labeled overlay is saved; no data are tracked by Git.

**Do not do:** train a model or design the dashboard.

## Day 2 — Full data audit and protocol freeze

**Goal:** know exactly what the data contain before writing model code.

- [ ] Acquire all three folds through one documented route.
- [ ] Generate file hashes and provenance record.
- [ ] Implement counts and checks in P3-02.
- [ ] Create per-class/per-tissue count tables.
- [ ] Inspect at least 24 stratified overlays, including rare/dead nuclei and crowded patches.
- [ ] Quantify overlap, empty masks, disconnected IDs, and duplicates.
- [ ] Write explicit handling rules for every anomaly.
- [ ] Confirm Fold 1 train, Fold 2 validation, Fold 3 locked test.
- [ ] Freeze hypotheses and test restrictions; tag `protocol-v1`.

**End-of-day evidence:** `data_audit.json`, audit grid, and frozen protocol exist.

**Decision gate:** if mask semantics are not clear, stop here and resolve them.

## Day 3 — Targets, loaders, and metric truth tests

**Goal:** prove the plumbing before optimization.

- [ ] Implement semantic, global-instance, type, and boundary target conversion.
- [ ] Implement inverse conversion to five-channel PanNuke masks.
- [ ] Pass the ground-truth round-trip test.
- [ ] Build spatially aligned augmentations and deterministic validation loader.
- [ ] Write overlay alignment tests for flips and rotations.
- [ ] Implement/wrap bPQ, mPQ, class PQ, and detection F1.
- [ ] Pass perfect/missed/extra/merge/split/wrong-class/empty synthetic cases.
- [ ] Run a one-sample end-to-end path through loader → identity prediction → official metrics.

**End-of-day evidence:** round trip is perfect; all metric truth tests pass.

**Do not do:** accept a metric because it “looks plausible.”

## Day 4 — Semantic baseline engineering

**Goal:** a debuggable training system, not a final score.

- [ ] Implement E1 six-class U-Net from configuration.
- [ ] Compute class weights from Fold 1 only and document the formula.
- [ ] Implement cross-entropy + soft-Dice loss.
- [ ] Add train/validation loops, AMP, checkpointing, CSV/JSON logs, and resume.
- [ ] Pass CPU forward/backward and CUDA forward/backward if available.
- [ ] Overfit one batch and save prediction progression.
- [ ] Run two epochs on 64–128 patches.
- [ ] Resume from the smoke checkpoint for one additional epoch.

**End-of-day evidence:** one-batch overfit and checkpoint-resume gates pass.

## Day 5 — Train and lock E1

**Goal:** finish the honest simple baseline.

- [ ] Train E1 on Fold 1 using the frozen budget.
- [ ] Monitor loss, per-class Dice, and representative validation overlays.
- [ ] If training fails, diagnose before changing multiple factors.
- [ ] Run the small predefined connected-component threshold/area grid on Fold 2.
- [ ] Select and freeze post-processing using validation bPQ, tie-break mPQ.
- [ ] Save validation summary and five clear merge failures.
- [ ] Record model parameters, runtime, peak memory, and checkpoint hash.

**End-of-day evidence:** reproducible E1 checkpoint and frozen validation-selected post-processing.

## Day 6 — Boundary-aware model

**Goal:** finish the single meaningful modeling extension.

- [ ] Add boundary channel and component losses.
- [ ] Visualize ground-truth boundaries for isolated and touching instances.
- [ ] Overfit one batch with all loss components logged.
- [ ] Train E2 under the same data, encoder, augmentations, and budget as E1.
- [ ] Confirm that both semantic and boundary outputs learn nontrivial predictions.
- [ ] Save the selected validation checkpoint under the declared rule.

**End-of-day evidence:** trained E2 checkpoint with comparable run metadata.

**Do not do:** add a transformer because a loss curve is imperfect.

## Day 7 — Watershed calibration and locked test

**Goal:** produce the main evidence without test leakage.

- [ ] Implement marker-controlled watershed and phenotype assignment.
- [ ] Pass synthetic touching-instance tests.
- [ ] Run the predefined validation grid and save every row.
- [ ] Freeze E2 thresholds; tag `test-ready-v1`.
- [ ] Run E1 and E2 once on Fold 3.
- [ ] Save overall bPQ/mPQ, class PQ, detection metrics, Dice, and runtime.
- [ ] Compute paired bootstrap intervals for E2–E1.
- [ ] Write a lab note before interpreting the result.

**End-of-day evidence:** immutable main result table and confidence intervals.

**Rule:** if E2 loses, report and analyze it. Do not quietly retune on Fold 3.

## Day 8 — Downstream reliability and failure analysis

**Goal:** turn a segmentation repo into a biomedical image-analysis study.

- [ ] Compute counts, proportions, MAE, normalized MAE, and bias per class.
- [ ] Compute matched-instance area/equivalent-diameter errors in pixels.
- [ ] Stratify metrics by tissue and density quartile.
- [ ] Compare Dice rankings with PQ/count-error rankings.
- [ ] Generate required main plots.
- [ ] Rank failures using predefined rules.
- [ ] Label at least ten worst cases with the frozen taxonomy.
- [ ] Write three evidence-backed findings and three limitations.

**End-of-day evidence:** final analysis tables, plots, and failure grid.

## Day 9 — Streamlit research demo

**Goal:** make results inspectable without hiding methodology.

- [ ] Select a small rule-based set of held-out examples: typical, dense, rare-class, best, and worst.
- [ ] Confirm redistribution/attribution requirements before packaging any data-derived asset.
- [ ] Build overlay toggles and class legend.
- [ ] Add true vs predicted counts and error display.
- [ ] Add morphology table and CSV export.
- [ ] Add model/config/checkpoint identifiers to the app.
- [ ] Add limitations and research-only notice.
- [ ] Test the app entirely on CPU from precomputed outputs.

**End-of-day evidence:** one-command dashboard demo with transparent export.

## Day 10 — Reproducibility and public release

**Goal:** a supervisor can audit the work in under ten minutes.

- [ ] Generate the exact package lock on the training platform.
- [ ] Add synthetic CPU end-to-end CI test.
- [ ] Re-run every README command from a fresh environment.
- [ ] Replace every `TBD` or remove the corresponding row/claim.
- [ ] Add main result, class/tissue analysis, representative examples, and failures to README.
- [ ] Complete the model card, data attribution, citation file, and limitations.
- [ ] Scan Git history for data, checkpoints, secrets, and large files.
- [ ] Ask one technically competent person to follow the quick start.
- [ ] Fix only release-blocking problems; log extensions separately.
- [ ] Create release `v0.1.0` and pin the repository.

**End-of-day evidence:** green CI, reproducible smoke path, complete public README, and tagged release.

## Daily status template

```text
Date:
Time spent:
Issue(s):
What I changed:
Evidence that passed:
What failed or surprised me:
Decision made and why:
Exact first task tomorrow:
Commit:
```

## If a day slips

Protect these in order: correct data handling, metric validity, one complete baseline, locked test, failure analysis, documentation. Cut stretch ablations and unrestricted user uploads first. Never cut the audit or replace missing results with claims.
