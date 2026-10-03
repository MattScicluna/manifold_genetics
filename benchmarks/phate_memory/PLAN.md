# PHATE memory benchmark — plan

Living plan for measuring and reducing the peak memory (and time) of the
manifold-genetics PHATE step at biobank scale. Written to be picked up by any
session or agent: read **Status** first, then the phase you are on.

## Why

Running All of Us v9 (101,142-sample fit set) on a 128 GB machine:

1. `embed --knn 500 --t 50 --n-components 3` was OOM-killed at "Transforming".
   Cause: `embed` defaults to **no landmarks** (`n_landmark=None`), so PHATE
   builds dense n × n operators — 101,142² × 8 B = 82 GB each. The 2-D pipeline
   run survived because the `subsample` preset sets `n_landmark=10000,
   random_landmarking=True` (`pipeline/configfile.py`).
2. `--embed-batch-size 20000` was also OOM-killed, on the first batch. Batching
   slices the fit data, so `phate.PHATE.transform` no longer recognises it as
   the fit data (`utils.matrix_is_equivalent`) and takes the **new-data** path:
   `graph.extend_to_data` runs a neighbour search from every batch row against
   all fit rows. graphtools starts at `knn × search_multiplier` = 500 × 6 =
   3,000 neighbours and widens 6× at a time, up to n/2, when `knn_max` is unset.
   So batching — meant to save memory — made the fit-sample transform cost more,
   and returns re-approximated coordinates instead of the fit embedding.

Question: which steps actually dominate peak memory and time, how does each
scale, and which changes (in manifold-genetics first, then in `phate` /
`graphtools`) reduce the peak without changing results?

## Setup

| | |
|---|---|
| Machine | a SLURM JupyterHub node: 8 CPUs, job memory cap 62.5 GiB (shared with the notebook) |
| Data | UKBB balanced fit-set PCA, 59,264 × 20 (`fit_pca_20.csv` from a `subsample` run); the project-set PCA, 486,748 × 20, from the same run |
| Code | a clean clone of `main` (8f2bb36) run with an existing environment, `PYTHONPATH` pointing at the clone's `src` |
| Versions | Python 3.10.16, phate 2.0.0, graphtools 2.1.0, scikit-learn 1.7.2, numpy 2.2.6 |
| Published settings | knn 500, t 50, decay 40, n_landmark 10,000, random landmarking |

Machine-specific paths live in `env.sh` next to the harness (not committed):
`PATH` (the Python environment), `PYTHONPATH`, `PCA` (fit-set CSV), `OUT`.
UKBB and AoU data never leave their environments; the harness records only
timings, memory and array shapes.

## Method

`bench_phate.py` runs one configuration per process (no carry-over) through the
real wrapper (`manifold_genetics.embeddings.phate.PHATE`): load → `fit` →
`transform`. For each step it records:

- **peak RSS of the step**: Linux `VmHWM`, reset before the step by writing `5`
  to `/proc/self/clear_refs`; a 50 ms RSS sampler cross-checks it. RSS includes
  native allocations (BLAS, sklearn's neighbour search) that `tracemalloc` misses.
- **time** (wall clock).
- **every `NearestNeighbors.kneighbors` call**: rows × n_neighbors, and the
  implied `rows × n_neighbors × 16 B` (float64 distances + int64 indices).
- a **ceiling guard** (default 56 GB): the run stops itself and records
  "exceeded" before the node's OOM killer can take the notebook with it.

`run_matrix.sh <phase>` runs a phase; `summarize.py results.jsonl` prints the table.

```bash
# on the compute node, from the directory holding the harness and env.sh
B=$PWD; source $B/env.sh
setsid nohup bash -c "cd $B && $B/run_matrix.sh phase1" > $B/phase1.log 2>&1 < /dev/null &
python $B/summarize.py $B/results.jsonl
```

## Theory to compare against (n = fit samples, L = landmarks, k = knn)

| Structure | Size | n=59,264 | n=101,142 |
|---|---|---|---|
| dense n × n (no landmarks; ~3 live at once in the smoke test) | 8n² | 28 GB (~84 GB ×3) | 82 GB (~245 GB ×3) |
| landmark transitions, dense n × L | 8nL | 4.7 GB (L=10k) | 8.1 GB |
| landmark potential / MDS, L × L | 8L² | 0.8 GB | 0.8 GB |
| first neighbour search, all n | 16 · n · 6k | 2.8 GB | 4.9 GB |
| first search per batch of b | 16 · b · 6k | 0.96 GB (b=20k) | |
| widest search per batch (n/2) | 16 · b · n/2 | 9.5 GB (b=20k) | 16 GB |

Smoke test (synthetic, n=6,000, k=50): no landmarks peaked at ~3 × 8n²;
with landmarks the fit-sample transform cost ~0.

## Phases

Each phase appends to `results.jsonl`; record the table under **Results**.

- **Phase 1 — current behaviour, published settings, fit-sample transform.**
  2-D and 3-D unbatched; 3-D batched at 20k / 10k / 5k.
  Expect: unbatched transform ≈ free (same-data path); batched transform larger
  than unbatched and slower, with widening kneighbors calls.
- **Phase 1L — landmarks.** L = 2,000 / 10,000 / none.
  Expect: none ≈ 3 × 8n² → exceeds the ceiling at n=59k; L scales the n × L term.
- **Phase 2 — true out-of-sample transform** (held-out rows, the only case
  batching is for): fit on 40k, transform 20k held-out; batch 0 / 10k / 5k / 2k
  × `knn_max` unset / 3,000 / 1,500.
  Expect: peak ∝ batch × widest search; `knn_max` bounds the widest search.
- **Phase 3 — scaling of fit with n**: 10k / 20k / 40k / all, to extrapolate to
  AoU's 101k and to any larger fit set.

## Candidate optimisations (to test, not yet made)

In manifold-genetics (no structural change):

- **O1 — never batch the fit samples.** In `PHATE.transform`, when the input is
  the fit data, call `model.transform` once (PHATE returns its embedding). Fixes
  memory and correctness (batched fit coordinates are re-approximations).
- **O2 — landmarks on by default for `embed`** at large n (or refuse/warn when
  n > ~20k with `n_landmark=None`), matching the presets.
- **O3 — expose `knn_max`** for out-of-sample projection, bounding the search.
  Changes the kernel's far tail: needs a fidelity check (below).
- **O4 — batch size from a memory budget** (like PCA's `--memory-gb`):
  b = budget / (16 · widest search), instead of a fixed count.

In `phate` / `graphtools` (owner can change):

- **P1 — bounded `extend_to_data`**: process query rows in chunks inside the
  radius-widening loop, so the widest search never materialises for all rows.
- **P2 — float32 for distances / transitions** where precision allows.
- **P3 — avoid holding several n × L / n × n copies at once** (identify from
  phase 1/1L which copies coexist).

Decide P* only after phases 1–3 show which step dominates.

## Fidelity checks (for anything that changes numbers)

Against the unmodified run on the same inputs and seed: Procrustes disparity of
the embeddings, k-NN preservation (fraction of each point's 30 nearest
neighbours kept), and max absolute coordinate difference. O1 and O4 should be
exact for the fit samples; O3, P1, P2 are approximations and need numbers.

## Decision rules

- Batching stays only where it lowers peak memory in phase 2 at equal or
  acceptable time; otherwise rethink (O4 / P1).
- An optimisation ships when it lowers the measured peak of the dominant step
  and passes the fidelity checks (or is exact).

## Status

- [x] Harness, runner, summariser; smoke-tested locally.
- [x] Root cause of the AoU 3-D OOMs identified (no landmarks; batching path).
- [ ] Phase 1 (running since 2026-10-03)
- [ ] Phase 1L
- [ ] Phase 2
- [ ] Phase 3
- [ ] Choose optimisations; implement O* on a branch with tests; re-run
- [ ] Consider P* in phate/graphtools

## Results

(fill in from `summarize.py`)
