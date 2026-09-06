# Streamlit dashboard

The dashboard presents the committed, locked Fold 3 evaluation without loading
raw PanNuke data or model checkpoints. It includes the final headline metrics,
tissue- and class-level comparisons, deterministic qualitative examples, and the
study protocol and limitations.

Run it from the repository root:

```bash
streamlit run app/streamlit_app.py
```

Required committed inputs:

- `reports/fold3_final_evaluation.json`
- `reports/fold3_qualitative_examples.json`
- `reports/figures/fold3_qualitative_examples.png`

The application is a research demonstration. It does not accept arbitrary
clinical images and does not provide diagnoses or validated biomarkers.
