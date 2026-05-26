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
    "10k_wbh": {
        "embedding": SCRIPT_DIR / "10k_WBH/outputs/phate_knn500_t100.csv",
        "labels": SCRIPT_DIR / "10k_WBH/data/fit_labels.csv",
        "n_samples": 10_000,
        "knn": 500,
        "t": 100,
    },
    "random": {
        "embedding": SCRIPT_DIR / "60k_random/outputs/phate_knn500_t100.csv",
        "labels": SCRIPT_DIR / "60k_random/data/labels.csv",
        "n_samples": 60_000,
        "knn": 500,
        "t": 100,
    },
    "geosketch": {
        "embedding": SCRIPT_DIR / "geosketch_phate/outputs/embeddings/phate_2d.csv",
        "labels": SCRIPT_DIR / "geosketch_phate/data/fit_labels.csv",
        "n_samples": 60_000,
        "knn": 500,
        "t": 100,
    },
    "geosketch_10kwbh_pca": {
        "embedding": SCRIPT_DIR / "geosketch_phate/outputs/phate_10kwbh_pca_knn500_t100.csv",
        "labels": SCRIPT_DIR / "geosketch_phate/data/fit_labels.csv",
        "n_samples": 60_000,
        "knn": 500,
        "t": 100,
    },
}

# ============================================================================
# Style
# ============================================================================

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.linewidth": 0.8,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)

# ============================================================================
# Helpers
# ============================================================================


def load_colormap(path: Path, label_col: str) -> dict:
    with open(path) as f:
        cmap = json.load(f)
    return dict(cmap[label_col])  # preserves insertion order


def plot_single_plot(
    plot_df: pd.DataFrame, color_map: dict, label_col: str, save_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10), dpi=600)

    # Layer 1: NaN values as lightgray background
    missing_mask = plot_df[label_col].isna()
    if missing_mask.any():
        ax.scatter(
            plot_df.loc[missing_mask, "dim_1"],
            plot_df.loc[missing_mask, "dim_2"],
            s=4.0,
            color="lightgray",
            rasterized=True,
            edgecolors="none",
            zorder=1,
        )

    # Layer 2: each group in reversed colormap order → first entry lands on top
    present_groups = [g for g in color_map if g in plot_df[label_col].values]
    for group in reversed(present_groups):
        mask = plot_df[label_col] == group
        ax.scatter(
            plot_df.loc[mask, "dim_1"],
            plot_df.loc[mask, "dim_2"],
            s=4.0,
            color=color_map[group],
            rasterized=True,
            edgecolors="none",
            zorder=2,
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

color_map = load_colormap(COLORMAP_PATH, LABEL_COL)

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

    save_path = FIGURES_DIR / (
        f"aou_phate_{sampler}_{cfg['n_samples']}" f"_knn{cfg['knn']}_t{cfg['t']}.pdf"
    )

    print(f"[{sampler}] Plotting {len(plot_df):,} points...")
    plot_single_plot(plot_df, color_map, LABEL_COL, save_path)

print("\nDone.")
