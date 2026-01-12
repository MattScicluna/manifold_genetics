#!/bin/bash
#
# UKBB 10K WB + 5K Irish Subset Pipeline
#
# This is a wrapper that calls the generic subset pipeline.
# It defines UKBB-specific paths and pipeline parameters.
#
# Usage:
#   bash examples/ukbb/10k_WB_5K_Irish/run_pipeline.sh
#

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ANSI colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Print header
echo ""
echo "========================================="
echo "  UKBB 10K WB + 5K Irish Pipeline"
echo "========================================="
echo ""

# Define UKBB-specific paths
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
FIT_PLINK="${DATA_DIR}/fit_subset"
PROJECT_PLINK="${DATA_DIR}/project_subset"
FIT_LABELS="${DATA_DIR}/fit_labels.csv"
PROJECT_LABELS="${DATA_DIR}/project_labels.csv"
COLORMAP="${PROJECT_ROOT}/examples/colormaps/ukbb.json"

# Get compute resources
THREADS="${SLURM_CPUS_PER_TASK:-4}"
NUM_GPUS="${SLURM_GPUS_ON_NODE:-}"

# Call generic subset pipeline
print_status "Calling generic subset pipeline..."
echo ""

# Note: "$@" passes through any additional arguments (e.g., --skip-admixture for testing)
bash "${PROJECT_ROOT}/examples/generic/subset/run_pipeline.sh" \
    --fit-plink "$FIT_PLINK" \
    --project-plink "$PROJECT_PLINK" \
    --fit-labels "$FIT_LABELS" \
    --project-labels "$PROJECT_LABELS" \
    --colormap "$COLORMAP" \
    --output-dir "$OUTPUT_DIR" \
    --n-pcs 20 \
    --k-min 2 --k-max 10 \
    --embedding phate \
    --knn 500 \
    --t 50 \
    --n-landmark 10000 \
    --random-landmarking \
    --embedding-input fit \
    --admixture-group-column self_described_ancestry \
    --threads "$THREADS" \
    --neuraladmixture-batch-size 400 \
    ${NUM_GPUS:+--num-gpus "$NUM_GPUS"} \
    --skip-metrics \
    "$@"

echo ""
print_success "UKBB 10K WB + 5K Irish pipeline complete!"
echo ""
