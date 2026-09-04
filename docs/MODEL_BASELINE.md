# Model baseline plan

## Why this baseline

A plain semantic U-Net is familiar and fast, but connected components merge touching nuclei. PanNuke is an instance-segmentation dataset, so a portfolio project should expose and test that limitation rather than reporting semantic Dice alone.

The controlled experiment therefore changes one modeling factor: whether the network learns an explicit instance-boundary target.

## Shared backbone and training budget

- Architecture family: U-Net.
- Encoder: ResNet-34, ImageNet initialization.
- Input: RGB, 256 × 256.
- Semantic classes: background plus five nucleus phenotypes.
- Default batch size: 8 on a 12–16 GB GPU; increase only after a memory test.
- Precision: automatic mixed precision on CUDA.
- Optimizer: AdamW.
- Maximum epochs: 50.
- Early stopping: patience 10 on the declared validation selection metric.
- Augmentation: horizontal/vertical flip, 90-degree rotations, modest color/stain perturbation; no geometric transform that is not applied identically to all targets.
- Initialization, augmentations, batch sampling, optimizer, schedule, and epoch budget are held constant between experiments.

Actual values belong in YAML and the resolved configuration must be saved with each run.

## E1 — semantic-only baseline

### Output

`[B, 6, H, W]` semantic logits.

### Loss

`L_sem = L_weighted_CE + L_soft_Dice`

- Compute class weights from training data only.
- Clip or smooth extreme inverse-frequency weights; record the formula.
- Exclude or include background in soft Dice deliberately and document the choice.

### Instance reconstruction

1. Softmax semantic logits.
2. Define foreground probability as `1 - P(background)` and apply the validation-selected foreground threshold.
3. Within foreground pixels, select the most probable positive phenotype.
4. Remove objects below a validation-selected area threshold.
5. Connected components on foreground.
6. Assign an instance phenotype using mean semantic probabilities inside the component.

Expected failure: adjacent nuclei merge into one object.

## E2 — boundary-aware U-Net

### Output

`[B, 7, H, W]`, split into six semantic logits and one boundary logit. A shared decoder with separate final channel interpretation is sufficient for the MVP; describe it accurately rather than calling it two independent decoders.

### Loss

`L_total = L_sem + λ_boundary × (L_BCE + L_boundary_Dice)`

Start with `λ_boundary = 1.0`; adjust once on validation only if loss scales are grossly mismatched. Do not search a large hyperparameter grid.

### Instance reconstruction

1. Estimate foreground from semantic probabilities.
2. Suppress pixels with high boundary probability to form interior seeds.
3. Clean seeds with a small-object rule selected on validation.
4. Compute a distance transform inside the foreground.
5. Apply marker-controlled watershed.
6. Remove implausibly small instances using the validation-frozen threshold.
7. Assign class by mean semantic probability across each instance.
8. Convert instances back to the five-channel PanNuke metric format.

## Minimal experiment matrix

| ID | Model | Post-processing | Purpose | Required? |
| --- | --- | --- | --- | --- |
| S0 | none | ground-truth round-trip | Prove format conversion and metrics | Yes |
| E1 | semantic U-Net | connected components | Honest simple baseline | Yes |
| E2 | semantic + boundary U-Net | watershed | Main proposed system | Yes |
| E2a | E2 checkpoint | connected components | Isolate post-processing contribution | Recommended |
| E2b | E2 checkpoint | watershed, no color augmentation | Small robustness ablation | Only if time allows |

Do not add transformers, ensembles, foundation models, or HoVer-Net training before S0–E2 and the error analysis are complete.

## Training sequence

1. Pass tensor-shape and target-range tests.
2. Overfit one batch to near-zero training loss.
3. Run two epochs on 64–128 patches and verify checkpoint/resume/logging.
4. Train E1.
5. Lock E1 post-processing on validation data.
6. Train E2 under the same budget.
7. Lock E2 post-processing on validation data.
8. Freeze all decisions and run the test evaluation.

## Compute profiles

### GPU available

Use the full plan with mixed precision. A free cloud GPU can be sufficient for 256 × 256 patches, but runtime must be measured rather than promised.

### No reliable GPU

- Complete the entire pipeline on a stratified development subset.
- Use pretrained inference only as a clearly labeled auxiliary demonstration.
- Do not claim a full-dataset trained model.
- Prioritize correct metrics, conversion, and error analysis over an undertrained “full” model.

## What to log

- git commit;
- resolved configuration;
- seed and deterministic flags;
- package/platform/GPU information;
- train/validation sample counts;
- class weights;
- epoch-level losses and metrics;
- best-checkpoint rule;
- parameter count;
- peak GPU memory and inference throughput;
- post-processing parameters.

Weights & Biases is optional. A local CSV/JSON log committed with the result summary is sufficient.
