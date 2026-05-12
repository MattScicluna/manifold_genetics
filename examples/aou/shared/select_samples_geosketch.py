#!/usr/bin/env python3
"""
Select a geometrically representative sample subset using geosketch.

Reads a PCA CSV (manylatents format: sample_id, dim_1, dim_2, ...) and a .fam
file, runs geometric sketching (Hie et al., 2019) in PCA space, and writes a
FID/IID sample list for PLINK extraction.

Reference:
    Hie B et al. Geometric Sketching Compactly Summarizes the Single-Cell
    Transcriptomic Landscape. Cell Systems 8, 483-493 (2019).

Examples:
    python select_samples_geosketch.py \
        --pca hgdp_1kgp_proj/outputs/pca/transform_pca_20.csv \
        --fam hgdp_1kgp_proj/data/project_subset.fam \
        --output geosketch_phate/data/fit_samples.txt \
        --n-samples 50000
"""

import argparse
import sys

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Geosketch-based sample selection from PCA coordinates"
    )
    parser.add_argument("--pca", required=True,
                        help="Path to PCA CSV (sample_id, dim_1, dim_2, ...)")
    parser.add_argument("--fam", required=True,
                        help="Path to .fam file for FID/IID mapping")
    parser.add_argument("--output", required=True,
                        help="Output sample list file (FID TAB IID, for PLINK --keep)")
    parser.add_argument("--n-samples", type=int, default=50000,
                        help="Number of samples to select (default: 50000)")
    parser.add_argument("--n-pcs", type=int, default=None,
                        help="Number of PCs to use for sketching (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    try:
        from geosketch import gs
    except ImportError:
        print("Error: geosketch is not installed.")
        print("Install with: uv add --optional geosketch 'geosketch>=1.3'")
        print("  or: pip install 'geosketch>=1.3'")
        sys.exit(1)

    # Read PCA CSV
    print(f"Loading PCA coordinates from: {args.pca}")
    pca_df = pd.read_csv(args.pca)

    if "sample_id" not in pca_df.columns:
        print(f"Error: 'sample_id' column not found in {args.pca}")
        print(f"  Available columns: {', '.join(pca_df.columns)}")
        sys.exit(1)

    # Read .fam file for FID/IID mapping
    print(f"Loading .fam file: {args.fam}")
    fam = pd.read_csv(args.fam, sep=r"\s+", header=None,
                      names=["FID", "IID", "PID", "MID", "Sex", "Phenotype"])
    iid_to_fid = dict(zip(fam["IID"].astype(str), fam["FID"].astype(str)))
    available_samples = set(fam["IID"].astype(str))

    # Filter PCA to samples present in .fam
    pca_df["sample_id"] = pca_df["sample_id"].astype(str)
    pca_df = pca_df[pca_df["sample_id"].isin(available_samples)].reset_index(drop=True)
    print(f"  Samples in PCA + .fam intersection: {len(pca_df)}")

    if len(pca_df) == 0:
        print("Error: No samples overlap between PCA CSV and .fam file.")
        sys.exit(1)

    # Build feature matrix
    dim_cols = [c for c in pca_df.columns if c != "sample_id"]
    if args.n_pcs is not None:
        dim_cols = dim_cols[:args.n_pcs]
    X = pca_df[dim_cols].values.astype(np.float32)
    print(f"  Using {len(dim_cols)} PCs for geometric sketching")

    # Run geosketch
    n = min(args.n_samples, len(pca_df))
    if n < args.n_samples:
        print(f"  Warning: requested {args.n_samples} samples but only {len(pca_df)} available")
    print(f"  Running geosketch: {len(pca_df)} -> {n} samples (seed={args.seed})")

    indices = gs(X, n, replace=False, seed=args.seed)
    indices = np.sort(np.asarray(indices))

    selected_ids = pca_df.iloc[indices]["sample_id"].tolist()
    print(f"  Selected {len(selected_ids)} samples")

    # Write FID/IID pairs for PLINK
    missing = [sid for sid in selected_ids if sid not in iid_to_fid]
    if missing:
        print(f"Error: {len(missing)} selected IDs not found in .fam FID/IID mapping.")
        sys.exit(1)

    with open(args.output, "w") as f:
        for sid in selected_ids:
            f.write(f"{iid_to_fid[sid]}\t{sid}\n")

    print(f"Saved fit sample list ({len(selected_ids)} samples): {args.output}")


if __name__ == "__main__":
    main()
