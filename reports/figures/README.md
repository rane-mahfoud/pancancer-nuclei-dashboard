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