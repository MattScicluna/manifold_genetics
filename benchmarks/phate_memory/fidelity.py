#!/usr/bin/env python
"""Compare saved transform coordinates with a reference run.

For each <label>.npy in --dir: Procrustes disparity against the reference
(0 = identical up to rotation, scale and translation), the fraction of each
point's k nearest neighbours that are kept, and the largest absolute coordinate
difference after Procrustes alignment. Runs where the data live; prints numbers
only.
"""

import argparse
import glob
import os

import numpy as np
from scipy.spatial import procrustes
from sklearn.neighbors import NearestNeighbors


def knn_kept(a, b, k):
    ia = NearestNeighbors(n_neighbors=k + 1).fit(a).kneighbors(a, return_distance=False)[:, 1:]
    ib = NearestNeighbors(n_neighbors=k + 1).fit(b).kneighbors(b, return_distance=False)[:, 1:]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(ia, ib)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--reference", required=True, help="label of the reference run")
    ap.add_argument("--k", type=int, default=30)
    a = ap.parse_args()

    ref = np.load(os.path.join(a.dir, f"{a.reference}.npy"))
    print(f"{'run':22} {'procrustes':>11} {'knn kept':>9} {'max |diff|':>11}")
    for path in sorted(glob.glob(os.path.join(a.dir, "*.npy"))):
        label = os.path.basename(path)[:-4]
        emb = np.load(path)
        if emb.shape != ref.shape:
            print(f"{label:22} shape {emb.shape} != {ref.shape}")
            continue
        m1, m2, disparity = procrustes(ref, emb)
        print(
            f"{label:22} {disparity:>11.2e} {knn_kept(ref, emb, a.k):>9.3f} "
            f"{np.abs(m1 - m2).max():>11.2e}"
        )


if __name__ == "__main__":
    main()
