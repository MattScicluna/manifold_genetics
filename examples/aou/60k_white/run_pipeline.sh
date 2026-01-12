#!/bin/bash
#
# AoU 60K White Subset Pipeline
#
# This script runs the complete analysis pipeline for AoU 60K White subset
# using the shared pipeline runner with subsample mode.
#
# Usage:
#   bash examples/aou/60k_white/run_pipeline.sh
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
echo "  AoU 60K White Subset Pipeline"
echo "========================================="
echo ""

# Define AoU-specific paths
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
FIT_PLINK="${DATA_DIR}/fit_subset"
PROJECT_PLINK="${DATA_DIR}/project_subset"
FIT_LABELS="${DATA_DIR}/fit_labels.csv"
PROJECT_LABELS="${DATA_DIR}/project_labels.csv"
COLORMAP="${PROJECT_ROOT}/examples/colormaps/aou.json"

# Step 1: Check if data exists, prepare if needed
print_status "Checking for processed data..."

if [[ ! -f "${FIT_PLINK}.bed" ]] || [[ ! -f "${PROJECT_PLINK}.bed" ]]; then
    print_warning "Processed data not found, running prepare_data.sh..."
    bash "${SCRIPT_DIR}/prepare_data.sh"
else
    print_success "Processed data found"
fi

# Step 2: Verify required files exist
print_status "Verifying required files..."

REQUIRED_FILES=(
    "${FIT_PLINK}.bed"
    "${FIT_PLINK}.bim"
    "${FIT_PLINK}.fam"
    "${PROJECT_PLINK}.bed"
    "${PROJECT_PLINK}.bim"
    "${PROJECT_PLINK}.fam"
    "${FIT_LABELS}"
    "${PROJECT_LABELS}"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        MISSING_FILES+=("$file")
    fi
done

if [[ ${#MISSING_FILES[@]} -gt 0 ]]; then
    echo "Missing required files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Please run prepare_data.sh first"
    exit 1
fi

print_success "All required files found"

# Step 3: Check virtual environment
print_status "Checking virtual environment..."

if [[ -z "$VIRTUAL_ENV" ]]; then
    print_warning "Virtual environment not activated"
    echo ""
    echo "Please activate the virtual environment first:"
    echo "  cd ${PROJECT_ROOT}"
    echo "  source .venv/bin/activate"
    echo ""
    exit 1
fi

print_success "Virtual environment active: $VIRTUAL_ENV"

# Step 4: Get data statistics
print_status "Getting data statistics..."

FIT_SAMPLES=$(wc -l < "${FIT_PLINK}.fam")
PROJECT_SAMPLES=$(wc -l < "${PROJECT_PLINK}.fam")
SNP_COUNT=$(wc -l < "${FIT_PLINK}.bim")

echo "  Fit dataset: $FIT_SAMPLES samples"
echo "  Project dataset: $PROJECT_SAMPLES samples"
echo "  SNPs: $SNP_COUNT"

# Step 5: Detect cluster environment
source "${PROJECT_ROOT}/examples/_shared/detect_cluster.sh"

# Step 6: Run shared pipeline with subsample mode
# Note: Uses --mode subsample (fit on subset, transform on subset by default)
# Gets landmarking by default - use --embedding-input both to project on full 60K
print_status "Running pipeline with subsample mode..."
echo ""

# Note: "$@" passes through any additional arguments (e.g., --skip-admixture for testing)
bash "${PROJECT_ROOT}/examples/_shared/run_pipeline.sh" \
    --mode subsample \
    --fit-plink "$FIT_PLINK" \
    --project-plink "$PROJECT_PLINK" \
    --fit-labels "$FIT_LABELS" \
    --project-labels "$PROJECT_LABELS" \
    --colormap "$COLORMAP" \
    --output "$OUTPUT_DIR" \
    --n-pcs 20 \
    --k-min 2 --k-max 10 \
    --embedding phate \
    --admixture-group-column race_ethnicity \
    --threads "$CLUSTER_CPUS" \
    ${CLUSTER_GPUS:+--num-gpus "$CLUSTER_GPUS"} \
    "$@"

# Summary
echo ""
print_success "Pipeline complete!"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "  - PCA: outputs/pca/"
echo "  - Admixture: outputs/admixture/"
echo "  - Embeddings: outputs/embeddings/"
echo "  - Figures: outputs/figures/"
echo ""
