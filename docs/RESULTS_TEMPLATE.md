# Results and experiment record template

Copy this file to `reports/run_<experiment_id>.md` for every full run.

## Run identity

- Experiment ID:
- Date/time UTC:
- Git commit:
- Configuration file:
- Resolved configuration artifact:
- Seed:
- Dataset source/revision:
- Fold assignment:
- Checkpoint SHA-256:
- Hardware:
- Python/PyTorch/CUDA:
- Training wall time:
- Peak GPU memory:

## Selection contract

- Checkpoint selection metric:
- Epoch selected:
- Post-processing search space:
- Winning validation parameters:
- Was any test metric observed before these choices were frozen? `No/Yes — explain`

## Overall metrics

| Split | bPQ | mPQ | Detection P | Detection R | Detection F1 | Foreground Dice | Inference ms/patch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Test | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Class-wise test metrics

| Class | Ground-truth instances | PQ | Detection F1 | Count MAE | Normalized MAE | Count bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Neoplastic | TBD | TBD | TBD | TBD | TBD | TBD |
| Inflammatory | TBD | TBD | TBD | TBD | TBD | TBD |
| Connective/soft tissue | TBD | TBD | TBD | TBD | TBD | TBD |
| Dead | TBD | TBD | TBD | TBD | TBD | TBD |
| Non-neoplastic epithelial | TBD | TBD | TBD | TBD | TBD | TBD |

## Main findings

1. TBD — support with a table/figure and uncertainty interval.
2. TBD — support with a table/figure and uncertainty interval.
3. TBD — support with a table/figure and uncertainty interval.

## Failures

| Patch ID | Tissue | Failure category | Evidence | Likely mechanism | Appropriate next experiment |
| --- | --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD | TBD |

## Deviations from protocol

List every deviation, when it occurred, whether test results had been observed, and its likely effect.

## Interpretation boundary

State what the run establishes and what it cannot establish. Do not call cell composition a biomarker without external clinical endpoints and validation.
