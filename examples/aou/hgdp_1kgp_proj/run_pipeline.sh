#!/bin/bash
#
# AoU-HGDP Cross-Projection Pipeline
#
# This script runs the complete cross-projection analysis:
# - HGDP+1KGP → AoU projection (HGDP as reference)
# - Uses subsample-like parameters due to AoU's large size
# - PCA (20 components)
# - Neural Admixture (K=2-10)
# - PHATE Embedding with landmarking
# - Separate visualizations for HGDP and AoU populations
#
# Usage:
#   bash examples/aou/hgdp_1kgp_proj/run_pipeline.sh
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
echo "  AoU-HGDP Cross-Projection Pipeline Analysis"
echo "================================================="
echo ""
echo "This will run the complete cross-projection analysis:"
echo "  - HGDP → AoU projection (HGDP as reference)"
echo "  - PCA (20 components)"
echo "  - Neural Admixture (K=2-10)"
echo "  - PHATE Embedding with landmarking (large dataset)"
echo "  - Separate visualizations for HGDP and AoU populations"
echo ""

# Define AoU-specific paths
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
FIT_PLINK="${DATA_DIR}/fit_subset"
PROJECT_PLINK="${DATA_DIR}/project_subset"
HGDP_LABELS="${DATA_DIR}/hgdp_labels.csv"
AOU_LABELS="${DATA_DIR}/aou_labels.csv"
HGDP_COLORMAP="${PROJECT_ROOT}/examples/colormaps/hgdp_1kgp.json"
AOU_COLORMAP="${PROJECT_ROOT}/examples/colormaps/aou.json"

# Step 1: Check if data exists
print_status "Checking for processed data..."

REQUIRED_FILES=(
    "${FIT_PLINK}.bed"
    "${FIT_PLINK}.bim"
    "${FIT_PLINK}.fam"
    "${PROJECT_PLINK}.bed"
    "${PROJECT_PLINK}.bim"
    "${PROJECT_PLINK}.fam"
    "${HGDP_LABELS}"
    "${AOU_LABELS}"
    "${HGDP_COLORMAP}"
    "${AOU_COLORMAP}"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        MISSING_FILES+=("$file")
    fi
done

if [[ ${#MISSING_FILES[@]} -gt 0 ]]; then
    print_warning "Processed data not found, running data preparation..."
    echo ""
    echo "Missing files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Please run prepare_data.sh first:"
    echo "  bash ${SCRIPT_DIR}/prepare_data.sh"
    exit 1
fi

print_success "All required files found"

# Step 2: Check virtual environment
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

# Step 3: Get data statistics
print_status "Getting data statistics..."

FIT_SAMPLES=$(wc -l < "${FIT_PLINK}.fam")
PROJECT_SAMPLES=$(wc -l < "${PROJECT_PLINK}.fam")
SNP_COUNT=$(wc -l < "${FIT_PLINK}.bim")

echo "  Fit dataset (HGDP): $FIT_SAMPLES samples"
echo "  Project dataset (AoU): $PROJECT_SAMPLES samples"
echo "  Common SNPs: $SNP_COUNT SNPs"

# Step 4: Detect cluster environment
source "${PROJECT_ROOT}/examples/_shared/detect_cluster.sh"

# Step 5: Run shared pipeline with projection mode
# Note: Uses --mode projection (fit on HGDP, transform on AoU)
# Overrides knn/t/landmarking for large dataset performance
print_status "Running cross-projection pipeline with landmarking..."
echo ""

# Note: "$@" passes through any additional arguments (e.g., --skip-admixture for testing)
bash "${PROJECT_ROOT}/examples/_shared/run_pipeline.sh" \
    --mode projection \
    --fit-plink "$FIT_PLINK" \
    --project-plink "$PROJECT_PLINK" \
    --fit-labels "$HGDP_LABELS" \
    --project-labels "$AOU_LABELS" \
    --fit-colormap "$HGDP_COLORMAP" \
    --project-colormap "$AOU_COLORMAP" \
    --output "$OUTPUT_DIR" \
    --n-pcs 20 \
    --k-min 2 --k-max 10 \
    --embedding phate \
    --knn 500 \
    --t 50 \
    --n-landmark 10000 \
    --random-landmarking \
    --admixture-group-column race_ethnicity \
    --neuraladmixture-batch-size 400 \
    --threads "$CLUSTER_CPUS" \
    ${CLUSTER_GPUS:+--num-gpus "$CLUSTER_GPUS"} \
    "$@"

# Summary
echo ""
echo "================================================="
print_success "Cross-projection pipeline analysis complete!"
echo "================================================="
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
echo "  📊 PCA (HGDP fit → AoU project):"
echo "    - pca/fit_pca_20.csv, pca/transform_pca_20.csv"
echo "    - figures/pca/"
echo ""
echo "  🧬 Admixture (K=2-10):"
echo "    - admixture/fit.{2..10}.csv, admixture/transform.{2..10}.csv"
echo "    - figures/admixture/"
echo ""
echo "  🗺️ PHATE Embedding:"
echo "    - embeddings/phate_2d.csv"
echo "    - figures/embeddings/"
echo ""
echo "Cross-projection workflow:"
echo "  • HGDP+1KGP used as diverse reference for training"
echo "  • AoU samples projected into HGDP-trained space"
echo "  • Landmarking used due to large AoU dataset size"
echo ""
