#!/bin/bash
#
# UKBB-HGDP Cross-Projection Pipeline
#
# This is a wrapper that calls the generic cross-projection pipeline.
# It defines UKBB-specific paths and pipeline parameters.
#
# Usage:
#   bash examples/ukbb/hgdp_1kgp_proj/run_pipeline.sh
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
echo "================================================="
echo "  UKBB-HGDP Cross-Projection Pipeline Analysis"
echo "================================================="
echo ""

# Define UKBB-specific paths
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
FIT_PLINK="${DATA_DIR}/fit_subset"
PROJECT_PLINK="${DATA_DIR}/project_subset"
HGDP_LABELS="${DATA_DIR}/fit_labels.csv"
UKBB_LABELS="${DATA_DIR}/project_labels.csv"
HGDP_COLORMAP="${PROJECT_ROOT}/examples/colormaps/hgdp_1kgp.json"
UKBB_COLORMAP="${PROJECT_ROOT}/examples/colormaps/ukbb.json"

# Get compute resources
THREADS="${SLURM_CPUS_PER_TASK:-4}"
NUM_GPUS="${SLURM_GPUS_ON_NODE:-}"

# Call generic cross-projection pipeline
print_status "Calling generic cross-projection pipeline..."
echo ""

# Note: "$@" passes through any additional arguments (e.g., --skip-admixture for testing)
bash "${PROJECT_ROOT}/examples/generic/hgdp_1kgp_proj/run_pipeline.sh" \
    --fit-plink "$FIT_PLINK" \
    --project-plink "$PROJECT_PLINK" \
    --fit-labels "$HGDP_LABELS" \
    --project-labels "$UKBB_LABELS" \
    --fit-colormap "$HGDP_COLORMAP" \
    --project-colormap "$UKBB_COLORMAP" \
    --output-dir "$OUTPUT_DIR" \
    --n-pcs 20 \
    --k-min 2 --k-max 10 \
    --embedding phate \
    --knn 100 \
    --t 3 \
    --admixture-group-column self_described_ancestry \
    --threads "$THREADS" \
    --neuraladmixture-batch-size 400 \
    --embed-batch-size 60000 \
    ${NUM_GPUS:+--num-gpus "$NUM_GPUS"} \
    --skip-metrics \
    "$@"

echo ""
print_success "UKBB-HGDP pipeline complete!"
echo ""
