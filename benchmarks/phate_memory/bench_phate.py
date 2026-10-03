#!/usr/bin/env python
"""Peak memory and time per step of a manifold-genetics PHATE embedding.

One configuration per process (the runner script starts a fresh one each time),
so peaks never carry over between configurations. Appends one JSON line per run.

Per-step peak: Linux keeps the process's peak resident set (VmHWM) in
/proc/self/status, and writing "5" to /proc/self/clear_refs resets it. So each
step is: reset -> run -> read VmHWM. A 50 ms RSS sampler runs alongside as a
cross-check and as the guard that stops a run before the node's OOM killer does.

Also records every sklearn kneighbors call (rows, n_neighbors), because the
batched path's cost is expected to be rows x n_neighbors x 16 bytes (float64
distances + int64 indices), with graphtools widening n_neighbors by 6x until the
kernel radius is covered -- up to n/2 when knn_max is unset.
"""

import argparse
import json
import os
import resource
import sys
import threading
import time

GB = 1024**3


# ---------------------------------------------------------------- memory probes
def _status_kb(field):
    if not os.path.exists("/proc/self/status"):  # macOS: smoke tests only
        import psutil

        mi = psutil.Process().memory_info()
        return int((mi.rss if field == "VmRSS" else _status_kb.peak) / 1024)
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return None


def rss_gb():
    return _status_kb("VmRSS") / 1024**2


def hwm_gb():
    return _status_kb("VmHWM") / 1024**2


_status_kb.peak = 0


def reset_hwm():
    if not os.path.exists("/proc/self/clear_refs"):
        return False
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
        return True
    except OSError:
        return False


class Sampler(threading.Thread):
    """RSS every 50 ms; tracks the peak since the last reset; kills the run at the ceiling."""

    def __init__(self, ceiling_gb, on_exceed):
        super().__init__(daemon=True)
        self.ceiling, self.on_exceed = ceiling_gb, on_exceed
        self.peak, self._stop = 0.0, threading.Event()

    def reset(self):
        self.peak = rss_gb()

    def run(self):
        while not self._stop.is_set():
            r = rss_gb()
            self.peak = max(self.peak, r)
            if r > self.ceiling:
                self.on_exceed(r)
            time.sleep(0.05)

    def stop(self):
        self._stop.set()


# ------------------------------------------------------------- kneighbors probe
KNN_CALLS = []


def instrument_kneighbors():
    from sklearn.neighbors import NearestNeighbors

    real = NearestNeighbors.kneighbors

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        rows = self.n_samples_fit_ if X is None else len(X)
        k = n_neighbors or self.n_neighbors
        t0 = time.time()
        out = real(self, X, n_neighbors, return_distance)
        KNN_CALLS.append(
            {
                "rows": int(rows),
                "n_neighbors": int(k),
                "expected_gb": rows * k * 16 / GB,
                "seconds": round(time.time() - t0, 2),
            }
        )
        return out

    NearestNeighbors.kneighbors = kneighbors


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca", required=True, help="PCA CSV (sample_id, dim_1..)")
    ap.add_argument("--n", type=int, default=0, help="rows to use (0 = all)")
    ap.add_argument("--n-pcs", type=int, default=20)
    ap.add_argument("--knn", type=int, default=500)
    ap.add_argument("--t", type=int, default=50)
    ap.add_argument("--n-components", type=int, default=3)
    ap.add_argument("--n-landmark", type=int, default=10000, help="0 = no landmarks (n x n)")
    ap.add_argument(
        "--random-landmarking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="default on, as the subsample preset (the published UKBB/AoU runs)",
    )
    ap.add_argument("--batch", type=int, default=0, help="embed_batch_size (0 = none)")
    ap.add_argument("--knn-max", type=int, default=0, help="phate knn_max (0 = unset)")
    ap.add_argument(
        "--transform-input",
        choices=["fit", "other"],
        default="fit",
        help="fit: transform the fit samples (what the pipeline does); "
        "other: transform held-out rows (true out-of-sample)",
    )
    ap.add_argument("--n-other", type=int, default=20000)
    ap.add_argument("--ceiling-gb", type=float, default=56)
    ap.add_argument("--label", default="")
    ap.add_argument(
        "--save-dir",
        default="",
        help="save transform coordinates as <label>.npy "
        "(for fidelity checks; keep inside the data's environment)",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    record = {
        "config": vars(a),
        "steps": {},
        "status": "ok",
        "host": os.uname().nodename,
        "cpus": os.cpu_count(),
    }

    def finish(status=None):
        if status:
            record["status"] = status
        record["kneighbors_calls"] = KNN_CALLS
        record["max_kneighbors_expected_gb"] = max((c["expected_gb"] for c in KNN_CALLS), default=0)
        record["process_peak_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        with open(a.out, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def exceeded(r):
        record["steps"].setdefault("_exceeded_at", {})["rss_gb"] = round(r, 2)
        finish(f"exceeded {a.ceiling_gb} GB")
        os._exit(3)

    sampler = Sampler(a.ceiling_gb, exceeded)
    sampler.start()
    hwm_ok = reset_hwm()
    record["hwm_reset_supported"] = hwm_ok

    def step(name, fn):
        reset_hwm()
        sampler.reset()
        base = rss_gb()
        calls_before = len(KNN_CALLS)
        t0 = time.time()
        out = fn()
        record["steps"][name] = {
            "seconds": round(time.time() - t0, 2),
            "rss_before_gb": round(base, 3),
            "peak_gb": round(hwm_gb() if hwm_ok else sampler.peak, 3),
            "sampled_peak_gb": round(sampler.peak, 3),
            "increase_gb": round((hwm_gb() if hwm_ok else sampler.peak) - base, 3),
            "kneighbors_calls": len(KNN_CALLS) - calls_before,
        }
        print(f"[{a.label}] {name}: {record['steps'][name]}", flush=True)
        return out

    instrument_kneighbors()
    import numpy as np
    import pandas as pd

    from manifold_genetics.embeddings.phate import PHATE

    def load():
        df = pd.read_csv(a.pca, dtype={"sample_id": str})
        cols = ["sample_id"] + [f"dim_{i}" for i in range(1, a.n_pcs + 1)]
        df = df[cols]
        rng = np.random.default_rng(0)
        order = rng.permutation(len(df))
        n_fit = len(df) if not a.n else min(a.n, len(df))
        fit = df.iloc[order[:n_fit]].reset_index(drop=True)
        other = df.iloc[order[n_fit : n_fit + a.n_other]].reset_index(drop=True)
        return fit, other

    fit_df, other_df = step("load", load)
    record["n_fit"] = len(fit_df)
    record["theory_gb"] = {
        "X_fit": len(fit_df) * a.n_pcs * 8 / GB,
        "landmark_transitions_dense": len(fit_df) * (a.n_landmark or len(fit_df)) * 8 / GB,
        "dense_n_by_n_if_no_landmarks": len(fit_df) ** 2 * 8 / GB,
        "first_search_full_n": len(fit_df) * a.knn * 6 * 16 / GB,
        "first_search_per_batch": (a.batch or len(fit_df)) * a.knn * 6 * 16 / GB,
        "widest_search_per_batch_n_over_2": (a.batch or len(fit_df)) * (len(fit_df) // 2) * 16 / GB,
    }

    model = PHATE(
        n_components=a.n_components,
        knn=a.knn,
        t=a.t,
        n_landmark=a.n_landmark or None,
        random_landmarking=bool(a.n_landmark) and a.random_landmarking,
        embed_batch_size=a.batch or None,
    )
    if a.knn_max:
        model.model.set_params(knn_max=a.knn_max)

    step("fit", lambda: model.fit(fit_df))
    target = fit_df if a.transform_input == "fit" else other_df
    record["n_transformed"] = len(target)
    emb = step("transform", lambda: model.transform(target))
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
        dims = [c for c in emb.columns if c != "sample_id"]
        np.save(os.path.join(a.save_dir, f"{a.label}.npy"), emb[dims].to_numpy())

    sampler.stop()
    finish()


if __name__ == "__main__":
    sys.exit(main())
