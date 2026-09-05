# Figure provenance

## `pannuke_fold1_sample_overlay.png`

This figure was generated from the first streamed sample of PanNuke Fold 1.

- Source: [PanNuke](https://arxiv.org/abs/2003.10778)
- Public mirror: [RationAI/PanNuke](https://huggingface.co/datasets/RationAI/PanNuke)
- Mirror revision: `1f498f7bd6a85ef5f204c592b41ac881eab61005`
- Tissue: Breast
- Transformation: original image displayed beside coloured ground-truth
  instance masks and class boundaries
- Dataset license: CC BY-NC-SA 4.0
- Contents: ground-truth annotation only; no model prediction

The repository's MIT code license does not replace or override the
dataset's CC BY-NC-SA 4.0 license.

## Metadata distribution figures

The following figures contain aggregate statistics derived from PanNuke:

- `pannuke_class_distribution.png`
- `pannuke_tissue_distribution.png`

They were generated from all 7,901 samples using the public mirror revision
`1f498f7bd6a85ef5f204c592b41ac881eab61005`.

- Source: [PanNuke](https://arxiv.org/abs/2003.10778)
- Public mirror: [RationAI/PanNuke](https://huggingface.co/datasets/RationAI/PanNuke)
- Dataset license: CC BY-NC-SA 4.0
- Contents: aggregate annotation counts; no model predictions

## `pannuke_annotation_anomalies.png`

This figure shows selected annotation-geometry cases discovered during the
complete PanNuke audit.

- Top row: examples of instance masks containing two disconnected components
- Bottom row: examples of pixels shared by more than one instance mask
- Overlap markers were enlarged to 3×3 pixels only for visual clarity
- Source: PanNuke Fold 1
- Mirror revision: `1f498f7bd6a85ef5f204c592b41ac881eab61005`
- Dataset license: CC BY-NC-SA 4.0
- Contents: ground-truth annotation inspection; no model predictions

## `tiny_batch_overfit.png`

This figure is a pipeline diagnostic showing whether the compact U-Net can
memorize two PanNuke Fold 1 training images.

- Split: Fold 1
- Sample indices: 0 and 1
- Training steps: 150
- Initial loss: 1.4332
- Final loss: 0.3991
- Loss reduction: 72.2%
- Purpose: pipeline verification only; not a model-performance result
- Dataset license: CC BY-NC-SA 4.0

## `unet_smoke_training_curve.png`

This figure records a short end-to-end semantic U-Net pipeline check.

- Training split: PanNuke Fold 1
- Validation split: PanNuke Fold 2
- Training samples: 32
- Validation samples: 16
- Epochs: 2
- Batch size: 2
- Best validation macro foreground Dice: 0.1659
- Fold 3 used: no
- Purpose: pipeline verification only; not a scientific performance result

## `unet_smoke_predictions.png`

This figure compares original PanNuke Fold 2 images, semantic ground truth,
and predictions from the smoke-test U-Net.

- Training data: 32 Fold 1 images
- Validation data: first Fold 2 samples
- Training epochs: 2
- Purpose: visual pipeline inspection only
- Interpretation: predictions remain noisy and are not scientific results
- Fold 3 used: no
- Dataset license: CC BY-NC-SA 4.0

## `unet_pilot_random_training_curve.png`

This figure records a reproducibly sampled semantic U-Net training pilot.

- Training split: PanNuke Fold 1
- Validation split: PanNuke Fold 2
- Training samples: 256 randomly selected samples
- Validation samples: 128 randomly selected samples
- Epochs: 5
- Batch size: 4
- Base channels: 16
- Best validation macro foreground Dice: 0.1065
- Fold 3 used: no
- Purpose: limited-data training diagnostic; not a final result

## `unet_pilot_random_training_curve.png`

This figure records an unweighted, reproducibly sampled semantic U-Net pilot.

- Training: 256 randomly selected Fold 1 images
- Validation: 128 randomly selected Fold 2 images
- Epochs: 5
- Best macro foreground Dice: 0.1065
- Fold 3 used: no
- Purpose: limited-data diagnostic, not a final result

## `unet_pilot_weighted_training_curve.png`

This figure records the matched class-balanced pilot.

- Training: the same 256 Fold 1 images
- Validation: the same 128 Fold 2 images
- Epochs: 5
- Best macro foreground Dice: 0.1661
- Only controlled change: logarithmic Fold 1 class weights
- Fold 3 used: no
- Purpose: loss-selection pilot, not a final result