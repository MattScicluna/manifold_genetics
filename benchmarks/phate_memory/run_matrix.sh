#!/bin/bash
# Run one phase of the PHATE memory benchmark. Each configuration is a fresh
# process (peaks never carry over). Results append to $OUT as JSON lines.
#
#   PCA=/path/to/ukbb_fit_pca_20.csv OUT=results.jsonl ./run_matrix.sh phase1
set -uo pipefail
: "${PCA:?set PCA to the fit-set PCA CSV}"
: "${OUT:=results.jsonl}"
: "${CEILING:=56}"
HERE="$(cd "$(dirname "$0")" && pwd)"
run() { python "$HERE/bench_phate.py" --pca "$PCA" --out "$OUT" --ceiling-gb "$CEILING" ${SAVE_DIR:+--save-dir "$SAVE_DIR"} "$@" || echo "run exited $? ($*)"; }

case "${1:-}" in
  phase1)  # published settings (knn 500, t 50, 10k random landmarks); fit-sample transform
    for nc in 2 3; do run --label "p1_nc${nc}_nobatch" --n-components $nc; done
    for b in 20000 10000 5000; do run --label "p1_nc3_b${b}" --n-components 3 --batch $b; done ;;
  phase1L) # landmark count: 2000, 10000, none (the `embed` CLI default)
    run --label "p1L_lm2000" --n-landmark 2000
    run --label "p1L_lm10000" --n-landmark 10000
    run --label "p1L_none" --n-landmark 0 ;;
  phase2)  # true out-of-sample transform: 20k held-out rows
    for b in 0 10000 5000 2000; do
      for km in 0 3000 1500; do
        run --label "p2_b${b}_km${km}" --transform-input other --n 40000 --batch $b --knn-max $km
      done
    done ;;
  phase3)  # scaling of fit with n
    for n in 10000 20000 40000 0; do run --label "p3_n${n}" --n $n; done ;;
  *) echo "usage: $0 phase1|phase1L|phase2|phase3"; exit 2 ;;
esac
