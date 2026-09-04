import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

INPUT_PATH = Path("reports/data_audit_metadata.json")
FIGURE_DIR = Path("reports/figures")

CLASS_ORDER = [
    "neoplastic",
    "inflammatory",
    "connective",
    "dead",
    "epithelial",
]

CLASS_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#0082c8",
    "#f58230",
    "#911eb4",
]

audit = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

class_counts = [audit["class_counts"][name] for name in CLASS_ORDER]
total_nuclei = audit["total_nuclei"]

figure, axis = plt.subplots(figsize=(10, 6))
bars = axis.bar(
    [name.title() for name in CLASS_ORDER],
    class_counts,
    color=CLASS_COLORS,
)

for bar, count in zip(bars, class_counts, strict=True):
    percentage = 100 * count / total_nuclei
    axis.text(
        bar.get_x() + bar.get_width() / 2,
        count,
        f"{count:,}\n({percentage:.1f}%)",
        ha="center",
        va="bottom",
    )

axis.set_title("PanNuke nucleus-class distribution")
axis.set_xlabel("Nucleus class")
axis.set_ylabel("Number of annotated nuclei")
axis.tick_params(axis="x", rotation=20)
figure.tight_layout()

class_output = FIGURE_DIR / "pannuke_class_distribution.png"
figure.savefig(class_output, dpi=200, bbox_inches="tight")
plt.close(figure)

tissue_items = sorted(
    audit["tissue_counts"].items(),
    key=lambda item: item[1],
)

tissue_names = [item[0] for item in tissue_items]
tissue_counts = [item[1] for item in tissue_items]

figure, axis = plt.subplots(figsize=(10, 9))
bars = axis.barh(tissue_names, tissue_counts, color="#4472c4")

for bar, count in zip(bars, tissue_counts, strict=True):
    axis.text(
        count,
        bar.get_y() + bar.get_height() / 2,
        f" {count:,}",
        va="center",
    )

axis.set_title("PanNuke patch distribution across tissue types")
axis.set_xlabel("Number of image patches")
axis.set_ylabel("Tissue type")
figure.tight_layout()

tissue_output = FIGURE_DIR / "pannuke_tissue_distribution.png"
figure.savefig(tissue_output, dpi=200, bbox_inches="tight")
plt.close(figure)

print("Saved:", class_output)
print("Saved:", tissue_output)
