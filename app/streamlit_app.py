"""Interactive dashboard for the locked PanNuke Fold 3 evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from pancancer_nuclei.dashboard import (
    class_comparison,
    detection_counts,
    headline_metrics,
    load_json_report,
    qualitative_comparison,
    tissue_comparison,
    validate_final_report,
)

ROOT = Path(__file__).resolve().parents[1]
FINAL_REPORT = ROOT / "reports" / "fold3_final_evaluation.json"
QUALITATIVE_REPORT = ROOT / "reports" / "fold3_qualitative_examples.json"
QUALITATIVE_FIGURE = ROOT / "reports" / "figures" / "fold3_qualitative_examples.png"

E1_COLOR = "#b39ddb"
E2_COLOR = "#d884a4"
TEXT_COLOR = "#4c3444"
GRID_COLOR = "#eadde5"
BACKGROUND_COLOR = "#fff8fb"
PANEL_COLOR = "#f8edf3"
CHANGE_CMAP = LinearSegmentedColormap.from_list(
    "lilac_rose_change",
    ("#d8c9ec", "#fff9fc", "#e7a9c0"),
)


st.set_page_config(
    page_title="PanCancer Nuclei Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_style() -> None:
    """Apply the project visual system to Streamlit components."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(145deg, {BACKGROUND_COLOR} 0%, #fff 58%,
                        #f7eff8 100%);
            color: {TEXT_COLOR};
        }}
        [data-testid="stSidebar"] {{
            background: #f5e8f0;
            border-right: 1px solid #e8d8e2;
        }}
        .block-container {{
            max-width: 1450px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }}
        .hero {{
            padding: 1.7rem 2rem;
            margin-bottom: 1.4rem;
            border: 1px solid #ead8e3;
            border-radius: 22px;
            background: linear-gradient(115deg, #f4dbe6 0%, #eee3f7 100%);
            box-shadow: 0 10px 28px rgba(92, 54, 77, 0.08);
        }}
        .hero h1 {{
            color: {TEXT_COLOR};
            font-size: clamp(2rem, 4vw, 3.5rem);
            line-height: 1.05;
            margin: 0 0 0.65rem 0;
            letter-spacing: -0.04em;
        }}
        .hero p {{
            color: #694e60;
            font-size: 1.05rem;
            margin: 0;
            max-width: 850px;
        }}
        .eyebrow {{
            color: #9a5270;
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }}
        .metric-card {{
            min-height: 145px;
            padding: 1.1rem 1.2rem;
            border: 1px solid #ead9e3;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.88);
            box-shadow: 0 6px 20px rgba(92, 54, 77, 0.06);
        }}
        .metric-label {{
            color: #765b6d;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .metric-value {{
            color: {TEXT_COLOR};
            font-size: 2.25rem;
            font-weight: 760;
            line-height: 1.15;
            margin: 0.45rem 0 0.25rem;
        }}
        .metric-delta {{
            color: #477b56;
            font-size: 0.92rem;
            font-weight: 650;
        }}
        .protocol-badge {{
            display: inline-block;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            color: #376744;
            background: #dfefdf;
            border: 1px solid #c7dfc8;
            font-size: 0.82rem;
            font-weight: 700;
        }}
        .callout {{
            padding: 1rem 1.15rem;
            border-left: 5px solid {E2_COLOR};
            border-radius: 10px;
            background: {PANEL_COLOR};
            color: {TEXT_COLOR};
            margin: 0.5rem 0 1rem;
        }}
        h1, h2, h3 {{ color: {TEXT_COLOR}; }}
        div[data-testid="stMetric"] {{
            border: 1px solid #ead9e3;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.82);
            padding: 0.8rem 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_dashboard_reports() -> tuple[dict, dict]:
    """Load the immutable reports used by the dashboard."""
    final_report = load_json_report(FINAL_REPORT)
    qualitative_report = load_json_report(QUALITATIVE_REPORT)
    validate_final_report(final_report)
    return final_report, qualitative_report


def metric_card(
    label: str,
    value: float,
    absolute_change: float,
    relative_change: float,
) -> None:
    """Render one headline metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value:.3f}</div>
          <div class="metric-delta">
            +{absolute_change:.3f} · +{relative_change:.1f}% vs E1
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def grouped_bar_chart(
    frame: pd.DataFrame,
    label_column: str,
    y_label: str,
    height: float = 4.5,
) -> plt.Figure:
    """Create a consistently styled vertical E1/E2 comparison."""
    labels = frame[label_column].tolist()
    positions = np.arange(len(frame))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, height))
    figure.patch.set_facecolor("none")
    axis.set_facecolor("none")
    axis.bar(
        positions - width / 2,
        frame["e1"],
        width,
        label="E1 · semantic baseline",
        color=E1_COLOR,
    )
    axis.bar(
        positions + width / 2,
        frame["e2"],
        width,
        label="E2 · boundary-aware hybrid",
        color=E2_COLOR,
    )
    axis.set_xticks(positions, labels, rotation=12, ha="right")
    axis.set_ylabel(y_label)
    maximum = float(frame[["e1", "e2"]].max().max())
    axis.set_ylim(0.0, max(0.6, maximum * 1.22))
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=1)
    axis.tick_params(colors=TEXT_COLOR)
    axis.yaxis.label.set_color(TEXT_COLOR)
    figure.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    return figure


def tissue_bar_chart(frame: pd.DataFrame, metric_label: str) -> plt.Figure:
    """Create a horizontal tissue comparison ordered by the selected rule."""
    positions = np.arange(len(frame))
    height = max(7.0, len(frame) * 0.43)
    figure, axis = plt.subplots(figsize=(10.5, height))
    figure.patch.set_facecolor("none")
    axis.set_facecolor("none")
    axis.barh(
        positions + 0.18,
        frame["e1"],
        0.35,
        label="E1",
        color=E1_COLOR,
    )
    axis.barh(
        positions - 0.18,
        frame["e2"],
        0.35,
        label="E2",
        color=E2_COLOR,
    )
    axis.set_yticks(positions, frame["tissue"])
    axis.invert_yaxis()
    axis.set_xlabel(metric_label)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=1)
    axis.tick_params(colors=TEXT_COLOR)
    axis.xaxis.label.set_color(TEXT_COLOR)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    return figure


def formatted_comparison(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Format a comparison table for display."""
    display = frame.rename(
        columns={
            "tissue": "Tissue",
            "class": "Nucleus class",
            "metric": "Metric",
            "e1": "E1",
            "e2": "E2",
            "absolute_change": "E2 − E1",
        }
    )
    return display.style.format(
        {
            "E1": "{:.3f}",
            "E2": "{:.3f}",
            "E2 − E1": "{:+.3f}",
        }
    ).background_gradient(
        subset=["E2 − E1"],
        cmap=CHANGE_CMAP,
        vmin=-0.12,
        vmax=0.12,
    )


def render_overview(final_report: dict) -> None:
    """Render locked headline results and detection counts."""
    metrics = headline_metrics(final_report)
    st.header("Locked Fold 3 results")
    st.markdown(
        '<span class="protocol-badge">✓ Parameters frozen on Fold 2</span>',
        unsafe_allow_html=True,
    )
    st.write("")
    columns = st.columns(3)
    for column, row in zip(columns, metrics.to_dict("records"), strict=True):
        with column:
            metric_card(
                row["metric"],
                row["e2"],
                row["absolute_change"],
                row["relative_change_percent"],
            )

    st.markdown(
        """
        <div class="callout">
        E2 improves final binary PQ by <strong>19.0%</strong> and multiclass PQ
        by <strong>18.8%</strong> on all 2,722 held-out Fold 3 patches. The test
        set was evaluated once after model and post-processing selection ended.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.pyplot(
        grouped_bar_chart(metrics, "metric", "Score", height=4.2),
        use_container_width=True,
    )

    st.subheader("Detection accounting")
    counts = detection_counts(final_report)
    count_columns = st.columns(3)
    for column, row in zip(count_columns, counts.to_dict("records"), strict=True):
        change = int(row["change"])
        delta_color = "normal" if row["measure"] == "Matched nuclei" else "inverse"
        with column:
            st.metric(
                row["measure"],
                f"{int(row['e2']):,}",
                f"{change:+,} vs E1",
                delta_color=delta_color,
            )


def render_tissues(final_report: dict) -> None:
    """Render interactive tissue-level PQ analysis."""
    st.header("Tissue explorer")
    controls = st.columns([1.2, 1.0, 1.8])
    with controls[0]:
        metric_label = st.selectbox(
            "Metric",
            ("Multiclass PQ", "Binary PQ"),
        )
    with controls[1]:
        order = st.selectbox(
            "Order tissues by",
            ("E2 performance", "E2 improvement", "Tissue name"),
        )

    metric_key = "multiclass_pq" if metric_label == "Multiclass PQ" else "binary_pq"
    frame = tissue_comparison(final_report, metric_key)
    if order == "E2 performance":
        frame = frame.sort_values("e2", ascending=False)
    elif order == "E2 improvement":
        frame = frame.sort_values("absolute_change", ascending=False)
    else:
        frame = frame.sort_values("tissue")

    improved = int((frame["absolute_change"] > 0).sum())
    with controls[2]:
        st.metric(
            "Tissues improved by E2",
            f"{improved} of {len(frame)}",
            help=f"Count based on {metric_label}.",
        )

    st.pyplot(
        tissue_bar_chart(frame, metric_label),
        use_container_width=True,
    )

    selected_tissue = st.selectbox(
        "Inspect one tissue",
        sorted(frame["tissue"].tolist()),
    )
    selected = frame.loc[frame["tissue"] == selected_tissue].iloc[0]
    tissue_columns = st.columns(3)
    tissue_columns[0].metric("E1", f"{selected['e1']:.3f}")
    tissue_columns[1].metric("E2", f"{selected['e2']:.3f}")
    tissue_columns[2].metric(
        "E2 − E1",
        f"{selected['absolute_change']:+.3f}",
    )
    st.dataframe(
        formatted_comparison(frame),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download tissue comparison CSV",
        frame.to_csv(index=False).encode("utf-8"),
        file_name=f"fold3_{metric_key}_by_tissue.csv",
        mime="text/csv",
    )


def render_classes(final_report: dict) -> None:
    """Render semantic and instance performance by nucleus class."""
    st.header("Nucleus-class analysis")
    semantic_tab, instance_tab = st.tabs(("Semantic segmentation", "Instance segmentation"))
    with semantic_tab:
        semantic = class_comparison(final_report, "semantic")
        st.pyplot(
            grouped_bar_chart(semantic, "class", "Dice", height=4.6),
            use_container_width=True,
        )
        st.dataframe(
            formatted_comparison(semantic),
            use_container_width=True,
            hide_index=True,
        )
    with instance_tab:
        instance = class_comparison(final_report, "instance")
        st.pyplot(
            grouped_bar_chart(
                instance,
                "class",
                "Panoptic Quality",
                height=4.6,
            ),
            use_container_width=True,
        )
        st.dataframe(
            formatted_comparison(instance),
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        "Dead nuclei are extremely rare in the Fold 1 training pixels. Their "
        "class-specific estimates should therefore be interpreted cautiously, "
        "even though E2 improves their instance PQ on locked Fold 3."
    )


def render_qualitative(qualitative_report: dict) -> None:
    """Render fixed qualitative examples and their sample-level scores."""
    st.header("Fixed qualitative examples")
    protocol = qualitative_report["selection_protocol"]
    st.markdown(
        f"""
        <div class="callout">
        These examples were selected with fixed seed <strong>{protocol["selection_seed"]}</strong>
        from six tissues declared before inference. Selection was independent of
        model performance, so the panel includes both E2 gains and a genuine E1 win.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if QUALITATIVE_FIGURE.exists():
        st.image(
            str(QUALITATIVE_FIGURE),
            caption=(
                "Original patch, reference instances, E1 predictions, and E2 "
                "predictions. Colours denote nucleus class."
            ),
            use_container_width=True,
        )
    else:
        st.warning("The committed qualitative figure could not be found.")

    frame = qualitative_comparison(qualitative_report)
    display = frame.rename(
        columns={
            "fold3_index": "Fold 3 index",
            "tissue": "Tissue",
            "ground_truth_nuclei": "Reference nuclei",
            "e1_nuclei": "E1 nuclei",
            "e2_nuclei": "E2 nuclei",
            "e1_binary_pq": "E1 bPQ",
            "e2_binary_pq": "E2 bPQ",
            "absolute_change": "E2 − E1",
        }
    )
    styled = display.style.format(
        {
            "E1 bPQ": "{:.3f}",
            "E2 bPQ": "{:.3f}",
            "E2 − E1": "{:+.3f}",
        }
    ).background_gradient(
        subset=["E2 − E1"],
        cmap=CHANGE_CMAP,
        vmin=-0.3,
        vmax=0.3,
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_protocol(final_report: dict) -> None:
    """Render the study design, parameters, and limitations."""
    st.header("Methods and reproducibility")
    st.subheader("Three-fold research protocol")
    fold_columns = st.columns(3)
    fold_content = (
        (
            "Fold 1 · Training",
            "Train E1 semantic U-Net and E2 boundary-aware multitask U-Net.",
        ),
        (
            "Fold 2 · Validation",
            "Select checkpoints and post-processing; investigate failures.",
        ),
        (
            "Fold 3 · Locked test",
            "Evaluate once with no parameter tuning or model selection.",
        ),
    )
    for column, (title, body) in zip(fold_columns, fold_content, strict=True):
        with column:
            st.markdown(f"### {title}")
            st.write(body)

    methods = final_report["methods"]
    st.subheader("Compared systems")
    method_columns = st.columns(2)
    with method_columns[0]:
        st.markdown("### E1 · Semantic baseline")
        st.write(methods["e1"]["postprocessing"])
        st.code(f"minimum_instance_area = {methods['e1']['minimum_instance_area']} pixels")
    with method_columns[1]:
        st.markdown("### E2 · Boundary-aware hybrid")
        st.write(methods["e2"]["postprocessing"])
        configuration = methods["e2"]["configuration"]
        st.code("\n".join(f"{key} = {value}" for key, value in configuration.items()))

    st.subheader("Scope and limitations")
    st.markdown(
        """
        - PanNuke patches do not establish whole-slide or patient-level performance.
        - Pixel-derived morphology is not a physical measurement without scale metadata.
        - Rare classes and heterogeneous tissues require stratified interpretation.
        - This is a research demonstration, not a diagnostic system or clinical biomarker.
        - Raw PanNuke data and trained checkpoints are not redistributed by the dashboard.
        """
    )
    st.caption(
        f"Dataset revision: {final_report['revision']} · "
        f"Held-out samples: {final_report['number_of_samples']:,}"
    )


def main() -> None:
    """Run the Streamlit application."""
    apply_style()
    try:
        final_report, qualitative_report = load_dashboard_reports()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        st.error(str(error))
        st.stop()

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Biomedical image analysis · PanNuke</div>
          <h1>Boundary-aware pan-cancer nuclei segmentation</h1>
          <p>
            A reproducible comparison of semantic connected components and a
            boundary-aware hybrid instance pipeline across 19 H&amp;E tissue types.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("🔬 Nuclei dashboard")
        page = st.radio(
            "Explore",
            (
                "Final overview",
                "Tissue explorer",
                "Class analysis",
                "Qualitative examples",
                "Methods & reproducibility",
            ),
        )
        st.divider()
        st.markdown(
            '<span class="protocol-badge">Locked Fold 3</span>',
            unsafe_allow_html=True,
        )
        st.caption("2,722 held-out patches · 19 tissues · no Fold 3 tuning")
        st.caption("Research use only · not for clinical diagnosis")

    if page == "Final overview":
        render_overview(final_report)
    elif page == "Tissue explorer":
        render_tissues(final_report)
    elif page == "Class analysis":
        render_classes(final_report)
    elif page == "Qualitative examples":
        render_qualitative(qualitative_report)
    else:
        render_protocol(final_report)

    st.divider()
    st.caption(
        "Rane Mahfoud · MSc Biotechnical Medical Systems and Technologies · "
        "Reproducible biomedical imaging portfolio"
    )


if __name__ == "__main__":
    main()
