"""
Generate publication-quality PHATE figures for AoU random and geosketch subsets.

Reads pre-computed embeddings from:
  - examples/aou/60k_random/outputs/phate_knn500_t100.csv
  - examples/aou/geosketch_phate/outputs/embeddings/phate_2d.csv

Saves PDFs to examples/aou/figures/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================================
# Paths
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR / "../.."
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORMAP_PATH = PROJECT_ROOT / "examples/colormaps/aou.json"
LABEL_COL = "race_ethnicity"

DATASETS = {
    "random": {
        "embedding": SCRIPT_DIR / "60k_random/outputs/phate_knn500_t100.csv",
        "labels":    SCRIPT_DIR / "60k_random/data/labels.csv",
        "n_samples": 60_000,
        "knn":       500,
        "t":         100,
    },
    "geosketch": {
        "embedding": SCRIPT_DIR / "geosketch_phate/outputs/embeddings/phate_2d.csv",
        "labels":    SCRIPT_DIR / "geosketch_phate/data/fit_labels.csv",
        "n_samples": 60_000,
        "knn":       500,
        "t":         100,
    },
}

# ============================================================================
# Style
# ============================================================================

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.linewidth': 0.8,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
})

# ============================================================================
# Helpers
# ============================================================================

def load_colormap(path: Path, label_col: str, fallback: str = "#CCCCCC") -> dict:
    with open(path) as f:
        cmap = json.load(f)
    return {k: v for k, v in cmap[label_col].items()}, fallback


def build_colors(labels_series: pd.Series, color_map: dict, fallback: str) -> list:
    return [color_map.get(val, fallback) for val in labels_series]


def plot_single_plot(plot_df: pd.DataFrame, colors: list, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10), dpi=600)
    ax.scatter(
        plot_df["dim_1"],
        plot_df["dim_2"],
        c=colors,
        s=4.0,
        rasterized=True,
        edgecolors="none",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    plt.savefig(save_path, dpi=600, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================================
# Main
# ============================================================================

color_map, fallback = load_colormap(COLORMAP_PATH, LABEL_COL)

for sampler, cfg in DATASETS.items():
    emb_path = cfg["embedding"]
    lbl_path = cfg["labels"]

    if not emb_path.exists():
        print(f"[{sampler}] Embedding not found, skipping: {emb_path}")
        continue
    if not lbl_path.exists():
        print(f"[{sampler}] Labels not found, skipping: {lbl_path}")
        continue

    print(f"\n[{sampler}] Loading data...")
    embedding = pd.read_csv(emb_path)
    embedding["sample_id"] = embedding["sample_id"].astype(str)

    labels = pd.read_csv(lbl_path)
    labels["sample_id"] = labels["sample_id"].astype(str)

    plot_df = embedding.merge(labels[["sample_id", LABEL_COL]], on="sample_id", how="left")
    plot_df[LABEL_COL] = plot_df[LABEL_COL].fillna("No information")

    colors = build_colors(plot_df[LABEL_COL], color_map, fallback)

    save_path = FIGURES_DIR / (
        f"aou_phate_{sampler}_{cfg['n_samples']}"
        f"_knn{cfg['knn']}_t{cfg['t']}.pdf"
    )

    print(f"[{sampler}] Plotting {len(plot_df):,} points...")
    plot_single_plot(plot_df, colors, save_path)

print("\nDone.")
