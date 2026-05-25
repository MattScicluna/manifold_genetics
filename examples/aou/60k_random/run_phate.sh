#!/bin/bash
#
# AoU Random 60K PHATE
#
# Runs PHATE (knn=500, t=100, n-landmark=2000) on a random 60K subset of AoU.
# The subset is created by prepare_data.py from the 10k_WBH projected PCA.
#
# Prerequisites:
#   python examples/aou/60k_random/prepare_data.py
#
# Usage:
#   bash examples/aou/60k_random/run_phate.sh
#   # or submit as a batch job:
#   bash examples/_shared/submit_batch.sh examples/aou/60k_random/run_phate.sh
#
#SBATCH --cpus-per-task=8
#SBATCH --account=ctb-hussinju
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=phate_60k_aou_random
#SBATCH --output=phate_60k_%j.out
#SBATCH --error=phate_60k_%j.err

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
COLORMAP="${PROJECT_ROOT}/examples/colormaps/aou.json"

source "${PROJECT_ROOT}/.venv/bin/activate"

mkdir -p "${OUTPUT_DIR}"

manifold-genetics embed \
    --input "${DATA_DIR}/pca_20.csv" \
    --output "${OUTPUT_DIR}/phate_knn500_t100.csv" \
    --method phate \
    --knn 500 \
    --t 100 \
    --n-landmark 2000

manifold-genetics plot \
    --input "${OUTPUT_DIR}/phate_knn500_t100.csv" \
    --labels "${DATA_DIR}/labels.csv" \
    --colormap "${COLORMAP}" \
    --output "${OUTPUT_DIR}/phate_knn500_t100_race_ethnicity.png"
