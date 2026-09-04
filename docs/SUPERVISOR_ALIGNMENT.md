# Supervisor-facing relevance

This file is private planning material until the project contains real results. Do not paste generic fit language into an email unchanged.

## David Juncker

Strongest connection: converting microscopy images into interpretable cell-level measurements relevant to cancer profiling and diagnostic bioengineering.

Evidence to point to once complete:

- instance-level segmentation rather than only semantic masks;
- cell-type composition exported per image;
- explicit propagation from segmentation error to quantitative measurements;
- transparent dashboard for inspecting examples and failure modes;
- careful boundary between research features and validated biomarkers.

Possible email sentence after results exist:

> Building on my MSc work in breast-cancer cell-image segmentation, I developed a reproducible PanNuke study that tests how nucleus instance-segmentation errors alter downstream cell-type counts across tissues; the public repository includes PQ-based evaluation, failure analysis, and an inspectable dashboard.

## Tal Arbel

Secondary connection: reliability across heterogeneous biomedical-image domains and failure-aware evaluation.

Evidence to point to:

- tissue-stratified generalization rather than one aggregate score;
- paired uncertainty intervals;
- predefined failure analysis;
- honest reporting of rare-class and density-dependent errors.

Do not call this an uncertainty-estimation project unless predictive uncertainty is actually implemented and evaluated.

## Link to the MSc story

The clean narrative is progression, not repetition:

1. MSc: breast-cancer cell-image segmentation and comparison of U-Net, StarDist, and Cellpose.
2. Portfolio project: reproducible multi-tissue instance segmentation plus phenotype/count reliability.
3. PhD direction: trustworthy quantitative biomedical imaging, domain shift, and clinically meaningful validation.

## Three bullets for an academic CV

Use only after replacing bracketed fields with measured facts:

- Built a reproducible PyTorch pipeline for five-class nucleus instance segmentation across 19 PanNuke tissue types, with controlled semantic-only and boundary-aware U-Net experiments.
- Evaluated held-out performance using bPQ/mPQ, paired bootstrap intervals, and tissue/class/density-stratified failure analysis; achieved [measured result, without exaggeration].
- Quantified propagation of segmentation errors into cell-count and pixel-scale morphology estimates and developed a Streamlit dashboard for auditable result inspection.
