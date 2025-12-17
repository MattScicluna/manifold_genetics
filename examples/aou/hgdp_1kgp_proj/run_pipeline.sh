#!/bin/bash
#
# Run AoU-HGDP cross-projection analysis pipeline
#
# This script runs the complete cross-projection analysis:
# - HGDP+1KGP → AoU projection (HGDP as reference)
# - PCA (20 components)
# - Neural Admixture (K=2-5)
# - PHATE Embedding
# - Separate visualizations for HGDP and AoU populations
#

set -e

# AOU specific
export SLURM_CPUS_PER_TASK=32

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
echo "  - PHATE Embedding"
echo "  - Separate visualizations for HGDP and AoU populations"
echo ""

# Paths
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
FIT_PLINK="${DATA_DIR}/fit_subset"
PROJECT_PLINK="${DATA_DIR}/project_subset"
HGDP_LABELS="${DATA_DIR}/hgdp_labels.csv"
AOU_LABELS="${DATA_DIR}/aou_labels.csv"
HGDP_COLORMAP="${PROJECT_ROOT}/examples/colormaps/hgdp_1kgp.json"
AOU_COLORMAP="${PROJECT_ROOT}/examples/colormaps/aou.json"
THREADS="${SLURM_CPUS_PER_TASK:-4}"
NUM_GPUS="${SLURM_GPUS_ON_NODE:-}"

# Check if processed data exists
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

# Check if virtual environment is activated
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

# Get data statistics
print_status "Getting data statistics..."

FIT_SAMPLES=$(wc -l < "${FIT_PLINK}.fam")
PROJECT_SAMPLES=$(wc -l < "${PROJECT_PLINK}.fam")
SNP_COUNT=$(wc -l < "${FIT_PLINK}.bim")

echo "  Fit dataset (HGDP): $FIT_SAMPLES samples"
echo "  Project dataset (AoU): $PROJECT_SAMPLES samples"
echo "  Common SNPs: $SNP_COUNT SNPs"

# Run the cross-projection pipeline
print_status "Running cross-projection analysis pipeline..."

echo ""
echo "Command:"
echo "manifold-genetics pipeline \\"
echo "    --fit-plink ${FIT_PLINK} \\"
echo "    --project-plink ${PROJECT_PLINK} \\"
echo "    --labels ${HGDP_LABELS} \\"
echo "    --colormap ${HGDP_COLORMAP} \\"
echo "    --fit-labels ${HGDP_LABELS} \\"
echo "    --project-labels ${AOU_LABELS} \\"
echo "    --fit-colormap ${HGDP_COLORMAP} \\"
echo "    --project-colormap ${AOU_COLORMAP} \\"
echo "    --output ${OUTPUT_DIR} \\"
echo "    --n-pcs 20 \\"
echo "    --k-min 2 --k-max 10 \\"
echo "    --embedding phate --knn 500 --t 50 --n-landmark 10000 --random-landmarking \\"
echo "    --skip-metrics \\"
echo "    --threads ${THREADS} \\"
echo "    --neuraladmixture-batch-size 400"
[ -n "$NUM_GPUS" ] && echo "    --num-gpus ${NUM_GPUS}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run pipeline
manifold-genetics pipeline \
    --fit-plink "$FIT_PLINK" \
    --project-plink "$PROJECT_PLINK" \
    --labels "$HGDP_LABELS" \
    --colormap "$HGDP_COLORMAP" \
    --fit-labels "$HGDP_LABELS" \
    --project-labels "$AOU_LABELS" \
    --fit-colormap "$HGDP_COLORMAP" \
    --project-colormap "$AOU_COLORMAP" \
    --output "$OUTPUT_DIR" \
    --n-pcs 20 \
    --k-min 2 --k-max 10 \
    --embedding phate --knn 500 --t 50 --n-landmark 10000 --random-landmarking \
    --admixture-group-column race_ethnicity \
    --skip-metrics \
    --threads "$THREADS" \
    --neuraladmixture-batch-size 400 \
    ${NUM_GPUS:+--num-gpus "$NUM_GPUS"}

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
echo "    - pca/fit_pca_20.csv"
echo "    - pca/transform_pca_20.csv"
echo ""
echo "  🧬 Admixture (HGDP fit → AoU project):"
echo "    - admixture/fit.{2..10}.csv"
echo "    - admixture/transform.{2..10}.csv"
echo ""
echo "  🗺️ PHATE Embedding:"
echo "    - embeddings/phate_2d.csv"
echo "    - figures/embeddings/phate_by_*.png"
echo ""
