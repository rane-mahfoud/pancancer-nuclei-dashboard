# GitHub issue backlog

Create milestones named `M0 Pipeline`, `M1 Baseline`, `M2 Analysis`, and `M3 Release`. Suggested labels: `data`, `model`, `evaluation`, `app`, `documentation`, `testing`, `priority:high`, `stretch`.

Do not open all stretch issues on Day 1. Create issues P3-00 through P3-19; keep P3-20 and P3-21 in the roadmap until the MVP is stable.

## P3-00 — Bootstrap the reproducible repository

**Milestone:** M0 Pipeline  
**Labels:** documentation, testing, priority:high  
**Depends on:** nothing

### Tasks

- Initialize Git and the public repository.
- Add folder structure, environment, package metadata, license, and Git ignore rules.
- Add pre-commit checks for formatting/linting.
- Add a minimal CI job for import and unit tests.
- Record the actual Python and CUDA platform used.

### Acceptance criteria

- Fresh clone installs using documented commands.
- `python -c "import pancancer_nuclei"` succeeds.
- `pytest -q` and lint checks pass.
- No data, weights, credentials, or absolute personal paths are tracked.

## P3-01 — Acquire PanNuke and record provenance

**Milestone:** M0 Pipeline  
**Labels:** data, priority:high  
**Depends on:** P3-00

### Tasks

- Read and record the dataset license.
- Download all three folds from the original source or one declared mirror.
- Record source URL, access date, distribution/revision, sizes, and SHA-256 hashes.
- Keep the archives outside Git.

### Acceptance criteria

- The three folds are locally present.
- A provenance manifest exists without secrets or signed URLs.
- The README and data guide cite the original PanNuke work.

## P3-02 — Implement the dataset audit

**Milestone:** M0 Pipeline  
**Labels:** data, testing, priority:high  
**Depends on:** P3-01

### Tasks

- Validate shapes, dtypes, ranges, and aligned row counts.
- Count patches/instances by fold, tissue, and class.
- Detect empty masks, overlaps, disconnected IDs, and duplicate images.
- Generate a stratified image/mask audit grid.

### Acceptance criteria

- `reports/data_audit.json` is machine-readable.
- Audit figure includes multiple tissues, common classes, and rare classes.
- Every anomaly has a count and explicit handling rule.

## P3-03 — Freeze the research and split protocol

**Milestone:** M0 Pipeline  
**Labels:** documentation, evaluation, priority:high  
**Depends on:** P3-02

### Tasks

- Confirm Fold 1 train, Fold 2 validation, Fold 3 locked test.
- Freeze hypotheses, primary metrics, threshold-selection rule, and failure taxonomy.
- Document prohibited test-set uses.

### Acceptance criteria

- Protocol contains no unresolved choice that could be made after viewing test performance.
- Git tag `protocol-v1` marks the frozen document.

## P3-04 — Convert PanNuke annotations to canonical targets

**Milestone:** M0 Pipeline  
**Labels:** data, testing, priority:high  
**Depends on:** P3-02

### Tasks

- Convert five positive channels to semantic, global-instance, type, and boundary targets.
- Implement the inverse conversion to the official five-channel metric format.
- Resolve overlapping positive pixels deterministically if any exist.

### Acceptance criteria

- Ground-truth round trip preserves all valid positive instances.
- Unit tests cover empty, single-instance, touching, and multi-class patches.
- Mapping order is asserted in code, not inferred from filenames.

## P3-05 — Implement transforms and data loaders

**Milestone:** M0 Pipeline  
**Labels:** data, testing, priority:high  
**Depends on:** P3-04

### Tasks

- Build train/validation/test datasets.
- Apply spatial transforms identically to all targets.
- Add intensity normalization and conservative H&E color augmentation.
- Add deterministic worker seeding.

### Acceptance criteria

- Overlay tests show perfect image/target alignment after transforms.
- Validation/test transforms are deterministic.
- A loader returns documented tensor shapes and dtypes.

## P3-06 — Validate PQ and detection metrics

**Milestone:** M0 Pipeline  
**Labels:** evaluation, testing, priority:high  
**Depends on:** P3-04

### Tasks

- Implement or wrap bPQ, mPQ, class-wise PQ, and detection metrics.
- Match official PanNuke semantics, including IoU > 0.5 matching.
- Create synthetic perfect/missed/extra/merge/split/wrong-class cases.

### Acceptance criteria

- All hand-calculated cases pass.
- Ground-truth round trip yields perfect metric results.
- Empty-class behavior is documented and tested.

## P3-07 — Build the semantic-only U-Net baseline

**Milestone:** M1 Baseline  
**Labels:** model, priority:high  
**Depends on:** P3-05

### Tasks

- Implement six-class U-Net with ResNet-34 encoder.
- Implement class-weighted cross-entropy plus soft Dice.
- Add parameter count and shape tests.

### Acceptance criteria

- Forward and backward passes succeed on CPU and CUDA when available.
- Loss is finite for normal and empty-foreground batches.
- Model configuration is fully represented in YAML.

## P3-08 — Pass overfit and smoke-training gates

**Milestone:** M1 Baseline  
**Labels:** model, testing, priority:high  
**Depends on:** P3-07

### Tasks

- Overfit one batch.
- Run a two-epoch subset experiment.
- Test checkpoint saving/resume and metric logging.

### Acceptance criteria

- One-batch loss decreases substantially and predictions visually approach targets.
- Resumed training reproduces the expected next epoch.
- Run directory contains config, seed, environment, metrics, and checkpoint hash.

## P3-09 — Train and evaluate E1 on validation data

**Milestone:** M1 Baseline  
**Labels:** model, evaluation, priority:high  
**Depends on:** P3-08, P3-06

### Tasks

- Train E1 under the frozen budget.
- Calibrate foreground and minimum-size rules on validation only.
- Produce validation tables and overlays.

### Acceptance criteria

- The selected checkpoint and post-processing rule are reproducible.
- Validation bPQ/mPQ and secondary metrics are saved.
- At least five merge failures are documented.

## P3-10 — Add the boundary target and output

**Milestone:** M1 Baseline  
**Labels:** model, data, testing, priority:high  
**Depends on:** P3-04, P3-07

### Tasks

- Add boundary-target generation to the dataset.
- Add the seventh model output and boundary loss.
- Visualize target and predicted boundaries.

### Acceptance criteria

- Boundary conventions are tested on isolated and touching nuclei.
- Total and component losses are logged separately.
- One-batch overfit succeeds.

## P3-11 — Implement watershed reconstruction

**Milestone:** M1 Baseline  
**Labels:** model, evaluation, priority:high  
**Depends on:** P3-10

### Tasks

- Create interior markers from foreground and boundary outputs.
- Apply marker-controlled watershed.
- Assign one phenotype to each predicted instance.
- Export official-format five-channel masks.

### Acceptance criteria

- Synthetic touching objects split correctly.
- Predicted instance IDs are unique and consecutive per patch.
- Output passes the same validation checks as ground truth.

## P3-12 — Train E2 and calibrate post-processing

**Milestone:** M1 Baseline  
**Labels:** model, evaluation, priority:high  
**Depends on:** P3-10, P3-11

### Tasks

- Train E2 under the same budget as E1.
- Run the predefined validation threshold grid.
- Freeze the best rule by validation bPQ, breaking ties with mPQ.

### Acceptance criteria

- Full grid is saved, not just the winning row.
- E2 vs E1 validation comparison is generated.
- Configuration and post-processing rule are tagged `test-ready-v1`.

## P3-13 — Run the locked test evaluation

**Milestone:** M2 Analysis  
**Labels:** evaluation, priority:high  
**Depends on:** P3-03, P3-06, P3-09, P3-12

### Tasks

- Run E1 and E2 once on Fold 3.
- Compute primary and secondary endpoints.
- Compute paired bootstrap confidence intervals.
- Archive immutable result manifests.

### Acceptance criteria

- Overall bPQ/mPQ, paired differences, and intervals are reported.
- No test-driven configuration change is made.
- Negative or null results remain visible.

## P3-14 — Analyze performance by class and tissue

**Milestone:** M2 Analysis  
**Labels:** evaluation, priority:high  
**Depends on:** P3-13

### Tasks

- Produce class-wise PQ and tissue-wise bPQ/mPQ.
- Flag small or absent class/tissue strata.
- Relate errors to training prevalence descriptively.

### Acceptance criteria

- Tables have denominators/sample counts.
- Macro and micro aggregation are not conflated.
- Plot colors/order are consistent across the report and dashboard.

## P3-15 — Quantify counting and morphometry error

**Milestone:** M2 Analysis  
**Labels:** evaluation, priority:high  
**Depends on:** P3-13

### Tasks

- Compute true/predicted per-class cell counts by patch.
- Report MAE, normalized MAE, and mean bias.
- Match true positives and quantify area/equivalent-diameter error.
- Stratify count error by density quartile.

### Acceptance criteria

- Outputs use pixels, not invented micrometers.
- Empty-class cases are handled explicitly.
- At least one example shows why Dice and count error differ.

## P3-16 — Perform predefined failure analysis

**Milestone:** M2 Analysis  
**Labels:** evaluation, documentation, priority:high  
**Depends on:** P3-13

### Tasks

- Rank patches by predefined failure scores.
- Label at least ten worst cases with the frozen taxonomy.
- Build a failure grid and short case table.

### Acceptance criteria

- Selection rule is described.
- Failures include merge, split, missed, spurious, and phenotype errors where present.
- Reference-label ambiguity is distinguished from model error.

## P3-17 — Build the precomputed-example dashboard

**Milestone:** M3 Release  
**Labels:** app, priority:high  
**Depends on:** P3-14, P3-15, P3-16

### Tasks

- Select a small, rule-based set of held-out examples.
- Add image/reference/prediction overlay controls.
- Add counts, proportions, morphology summaries, and CSV export.
- Add research-only and data-license notices.

### Acceptance criteria

- App starts with one command on CPU.
- It does not require the full dataset or training checkpoint.
- Exported rows contain instance ID, predicted class, area, equivalent diameter, and confidence when available.

## P3-18 — Harden reproducibility and CI

**Milestone:** M3 Release  
**Labels:** testing, documentation, priority:high  
**Depends on:** P3-13

### Tasks

- Generate an exact lock file on the actual training platform.
- Add a CPU end-to-end smoke test using synthetic data.
- Add lint/test CI and validate every documented command.
- Check repository history for large files or secrets.

### Acceptance criteria

- Fresh-environment smoke test passes.
- CI is green.
- No raw data, checkpoint, secret, or machine-specific path is tracked.

## P3-19 — Write the research-grade README and model card

**Milestone:** M3 Release  
**Labels:** documentation, priority:high  
**Depends on:** P3-13 through P3-18

### Tasks

- Replace every `TBD` with a measured result or remove the claim.
- Add main result table, stratified plots, failure cases, limitations, and exact commands.
- Add a model card and citations.
- Ask one technically competent reader to reproduce the smoke path.

### Acceptance criteria

- No unsupported novelty or clinical claim appears.
- Every figure is readable on mobile and has a caption.
- Repository landing page explains question, method, evidence, failures, and reproduction in under three minutes.

## P3-20 — Repeat across fold rotations

**Milestone:** stretch  
**Labels:** evaluation, stretch  
**Depends on:** P3-19

Freeze hyperparameters from the first protocol, repeat with held-out folds 1 and 2, and report mean ± standard deviation. Do not retune each rotation.

## P3-21 — Add CoNIC external validation

**Milestone:** stretch  
**Labels:** data, evaluation, stretch  
**Depends on:** P3-19

Create an explicit label-harmonization protocol before running anything. CoNIC is colon-specific and has a different six-class ontology; report detection/instance transfer separately from phenotype transfer.
