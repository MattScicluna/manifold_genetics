"""Sample 60K rows from AoU PCA output and create matching labels."""

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR / "../../.."

SRC_PCA = PROJECT_ROOT / "examples/aou/10k_WBH/outputs/pca/transform_pca_20.csv"
SRC_LABELS = PROJECT_ROOT / "examples/aou/10k_WBH/data/project_labels.csv"
OUT_DIR = SCRIPT_DIR / "data"

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading PCA CSV...")
pca = pd.read_csv(SRC_PCA)
print(f"  Total samples: {len(pca)}")

subset = pca.sample(n=60_000, random_state=42)
pca_out = OUT_DIR / "pca_20.csv"
subset.to_csv(pca_out, index=False)
print(f"Saved 60K PCA subset to {pca_out}")

print("Loading labels CSV...")
labels = pd.read_csv(SRC_LABELS)
labels["sample_id"] = labels["sample_id"].astype(str)
subset_ids = set(subset["sample_id"].astype(str))
labels_subset = labels[labels["sample_id"].isin(subset_ids)]
labels_out = OUT_DIR / "labels.csv"
labels_subset.to_csv(labels_out, index=False)
print(f"Saved {len(labels_subset)} label rows to {labels_out}")
print(f"  ({len(subset_ids) - len(labels_subset)} sample_ids had no label match)")
