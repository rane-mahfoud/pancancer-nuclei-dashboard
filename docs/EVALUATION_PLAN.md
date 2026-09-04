# Evaluation plan

## Evaluation principle

PanNuke provides instance masks and cell phenotypes. Semantic Dice alone cannot tell whether touching nuclei were correctly separated, so instance-level panoptic quality is the primary endpoint.

## Primary metrics

### Binary panoptic quality (bPQ)

All positive nuclei are treated as one class. Predicted and true instances are matched at IoU greater than 0.5, consistent with the PanNuke benchmark.

`PQ = detection quality × segmentation quality`

bPQ answers: were nuclei individually detected and delineated, regardless of phenotype?

### Multiclass panoptic quality (mPQ)

PQ is computed separately for each positive nucleus class and averaged. Report:

- class-wise PQ;
- macro average across the five classes;
- tissue-wise mPQ;
- macro average across tissue types.

The final README must state exactly how absent classes in a patch/tissue are handled. Match the official PanNuke implementation for benchmark tables and add tests for edge cases.

## Secondary metrics

| Metric | Level | Why it is included |
| --- | --- | --- |
| Foreground Dice | pixel | Connects to common segmentation literature but is not the main claim |
| Per-class semantic Dice | pixel/class | Diagnoses phenotype/background confusion |
| Detection precision, recall, F1 | instance | Separates missed and spurious nuclei |
| Count MAE | patch/class | Measures downstream composition error in intuitive units |
| Normalized MAE | patch/class | Makes common and rare classes more comparable |
| Mean count bias | patch/class | Shows systematic over- or under-counting |
| Matched-instance area error | instance | Tests morphometry reliability |
| Equivalent-diameter error | instance | Adds interpretable morphology without claiming physical units |
| Inference time and parameter count | system | Documents practicality |

R² and correlation may be plotted as descriptive supplements, but they must not replace absolute-error and bias reporting.

## Required stratification

1. Overall test set.
2. Each of the five nucleus classes.
3. Each tissue type with sufficient ground truth; flag unstable small strata.
4. Ground-truth nucleus-density quartiles.
5. Ground-truth instance-area quartiles for matched nuclei.

Optional: stain/color summary strata, only if the method is defined without looking at test outcomes.

## Statistical reporting

- Report the point estimate and a 95% bootstrap confidence interval for primary aggregate metrics.
- Bootstrap at the patch level unless a stronger grouping identifier is available.
- Use a fixed bootstrap seed and at least 1,000 resamples.
- For E2 minus E1, use paired resampling of the same patches.
- Be explicit that patch-level bootstrap does not capture patient/site clustering if those identifiers are unavailable.
- Do not run many null-hypothesis tests merely to obtain p-values; effect sizes and intervals are more useful here.

## Threshold-selection protocol

Tune only on Fold 2:

- foreground probability or class decision rule;
- boundary threshold;
- seed threshold/minimum seed area;
- minimum final instance area;
- watershed compactness, if nonzero.

Use a small, predefined search. Select the setting with the best validation bPQ, break ties with mPQ, then freeze it. Save the complete validation grid to CSV.

## Metric validation

Before evaluating predictions, create synthetic cases with known behavior:

- perfect match → PQ = 1;
- one missed object → recall and DQ decrease;
- one extra object → precision and DQ decrease;
- two true touching nuclei merged into one prediction;
- one true nucleus split into two predictions;
- correct instance with wrong phenotype;
- empty true/empty predicted patch;
- empty true/non-empty predicted patch.

Also perform a ground-truth round trip:

1. convert original five positive mask channels to canonical maps;
2. convert canonical maps back to official five-channel format;
3. evaluate the reconstruction against the original;
4. require perfect results except for explicitly documented malformed annotations.

## Locked-test rule

The test fold may be opened for file integrity and shape checks. Its performance must not be examined until:

- model architecture is frozen;
- training budget and checkpoint selection rule are frozen;
- post-processing thresholds are frozen;
- metric tests pass;
- required plots and failure categories are predefined.

After the first test run, report the result. Any later model change must be labeled exploratory and must not replace the original result silently.

## Failure analysis

Automatically rank patches by bPQ, mPQ, absolute count error, and merge/split indicators. Manually label at least ten worst cases using this taxonomy:

- missed faint nucleus;
- spurious stain/artifact detection;
- touching-nucleus merge;
- over-segmentation/split;
- phenotype confusion;
- rare/small dead-cell failure;
- border-truncated instance;
- dense cluster failure;
- ambiguous or potentially noisy reference label.

Show successes too: at least five representative patches selected by a rule set before viewing aesthetics, not only hand-picked best images.

## Required result files

```text
reports/
├── metrics/
│   ├── overall.json
│   ├── by_class.csv
│   ├── by_tissue.csv
│   ├── by_density_quartile.csv
│   ├── count_errors.csv
│   └── bootstrap_intervals.csv
├── predictions/
│   └── manifest.csv
├── figures/
│   ├── main_comparison.png
│   ├── pq_by_class.png
│   ├── pq_by_tissue.png
│   ├── count_agreement.png
│   └── failure_grid.png
└── model_card.md
```

Large prediction arrays remain outside Git; commit manifests, summary tables, compact figures, and a small legally compliant demo set only.
