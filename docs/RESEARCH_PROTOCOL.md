# Research protocol

## Working title

**Boundary-aware pan-cancer nucleus profiling: reliability of instance segmentation and cell-composition estimates across tissue types**

## Research question

How reliably can a lightweight boundary-aware U-Net recover nucleus instances and phenotypes across heterogeneous PanNuke tissues, and how do its errors affect downstream cell-composition measurements?

## Claims this study may support

- A controlled boundary-output ablation changes instance-level segmentation and counting performance relative to a semantic-only baseline.
- Performance varies across nucleus classes, tissue types, and cell-density regimes.
- Pixel overlap alone does not fully describe the reliability of downstream cell counts.

## Claims this study may not support

- clinical diagnostic, prognostic, or treatment utility;
- a validated cancer biomarker;
- patient-level generalization;
- superiority to state-of-the-art methods unless they are reproduced under a directly comparable protocol;
- biological conclusions outside the PanNuke annotation schema.

## Hypotheses

### H1 — instance separation

Adding a boundary target and marker-controlled watershed will improve validation and test bPQ relative to connected components applied to a semantic-only U-Net.

### H2 — rare-class reliability

The dead-cell class will have the lowest class-wise PQ and the largest normalized count error because it is rare and its instances are small.

### H3 — density sensitivity

Patches in the highest ground-truth nucleus-density quartile will show more merge errors and worse count MAE than patches in the lowest-density quartile.

### H4 — metric mismatch

Foreground Dice and count accuracy will not rank all tissue/class strata identically; strong pixel overlap can coexist with biased instance counts.

## Experimental units and split

- Experimental unit for model input: one 256 × 256 PanNuke patch.
- MVP split: Fold 1 training, Fold 2 validation, Fold 3 locked test.
- Hyperparameters, thresholds, and failure taxonomy are chosen using training/validation only.
- Fold 3 is evaluated once for the final report after the configuration is frozen.
- Optional extension: repeat the fixed protocol under two additional fold rotations without further tuning and report mean ± standard deviation.

The README must name the exact fold assignment. Do not casually call a single split “cross-validation.”

## Inputs and targets

### Input

- RGB H&E patch normalized to `[0, 1]`, followed by ImageNet normalization only if pretrained encoder weights are used.

### Semantic target

- Class 0: background.
- Class 1: neoplastic.
- Class 2: inflammatory.
- Class 3: connective/soft tissue.
- Class 4: dead.
- Class 5: non-neoplastic epithelial.

### Instance target

- Each positive PanNuke channel contains integer instance IDs local to that class.
- For class-agnostic instance operations, IDs must be relabeled to globally unique consecutive integers per patch.

### Boundary target

- Generate a one-pixel inner boundary from the globally unique instance map.
- Store the generation parameters in the resolved run configuration.
- Confirm visually on crowded and isolated examples before training.

## Models

### E1: semantic-only baseline

- U-Net with ResNet-34 encoder and ImageNet initialization.
- Six semantic output channels.
- Loss: weighted cross-entropy plus foreground multiclass soft Dice.
- Instance reconstruction: threshold foreground, remove tiny objects, connected components, then assign one phenotype per instance by majority semantic probability.

### E2: boundary-aware model

- Same U-Net encoder, decoder capacity, initialization, augmentations, and training budget.
- Six semantic channels plus one boundary channel.
- Loss: E1 loss plus weighted binary cross-entropy/Dice boundary loss.
- Instance reconstruction: foreground and boundary probabilities followed by marker-controlled watershed; phenotype assigned by mean per-pixel class probability inside each instance.

Using the same backbone makes the comparison interpretable. Do not replace E2 with a much larger model.

## Training controls

- Fix a primary seed before full training; add two confirmatory seeds only if time/compute allow.
- Mixed precision on CUDA.
- AdamW, learning-rate schedule, epoch budget, early-stopping rule, and class-weight method specified in YAML.
- Thresholds and minimum-object size calibrated only on Fold 2.
- Save the checkpoint with best validation mPQ or, if mPQ is too slow each epoch, best validation loss plus periodic mPQ checks. State which rule was used.
- Run a one-batch overfit test before the first full run.

## Planned analysis

1. Dataset and label audit.
2. E1 vs E2 main comparison.
3. Class-wise and tissue-wise PQ.
4. Density quartile analysis.
5. Per-class count MAE, normalized MAE, bias, and scatter plots.
6. Merge/split/missed/spurious/misclassified failure review on predefined worst cases.
7. Area and equivalent-diameter error for matched true-positive instances, expressed in pixels unless valid scale metadata are available.
8. Runtime and parameter count.

## Stop rules

- If a one-batch overfit test fails, stop full training and debug targets/loss.
- If official and local PQ disagree on synthetic unit cases, stop and repair metric conversion.
- If a data fold contains overlapping positive-class masks, quantify and define a deterministic resolution before training.
- If E2 fails to beat E1 on validation bPQ, still report it; do not tune on the test fold or hide the negative result.
- If the full dashboard threatens the analysis deadline, ship a precomputed-example dashboard rather than an unrestricted upload workflow.

## Definition of done

The project is done when code, configuration, data instructions, test results, figures, limitations, and dashboard can be examined independently. A trained checkpoint without the analysis is not completion.
