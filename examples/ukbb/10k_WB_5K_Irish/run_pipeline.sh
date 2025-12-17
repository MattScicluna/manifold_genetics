#!/bin/bash
#
# Run UKBB 10K WB + 5K Irish analysis pipeline
#
# This script runs the complete analysis pipeline:
# 1. PCA (20 components)
# 2. Neural Admixture (K=2-10)
# 3. PHATE Embedding
# 4. Visualization
# 5. Metrics (optional)
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

# Paths
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
FIT_PLINK="${DATA_DIR}/fit_subset"
PROJECT_PLINK="${DATA_DIR}/project_subset"
FIT_LABELS="${DATA_DIR}/fit_labels.csv"
PROJECT_LABELS="${DATA_DIR}/project_labels.csv"
COLORMAP="${PROJECT_ROOT}/examples/colormaps/ukbb.json"
THREADS="${SLURM_CPUS_PER_TASK:-4}"
NUM_GPUS="${SLURM_GPUS_ON_NODE:-}"

# Step 1: Check if data exists
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

# Step 4: Run the pipeline
print_status "Running analysis pipeline..."

echo ""
echo "Configuration:"
echo "  Fit PLINK: ${FIT_PLINK}"
echo "  Project PLINK: ${PROJECT_PLINK}"
echo "  Fit labels: ${FIT_LABELS}"
echo "  Project labels: ${PROJECT_LABELS}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  PCs: 20"
echo "  K range: 2-10"
echo "  Embedding: PHATE (knn=100)"
echo "  Threads: ${THREADS}"
if [[ -n "$NUM_GPUS" ]]; then
    echo "  GPUs: ${NUM_GPUS}"
fi
echo ""

# Build GPU args if available
GPU_ARGS=""
if [[ -n "$NUM_GPUS" ]] && [[ "$NUM_GPUS" -gt 0 ]]; then
    GPU_ARGS="--num-gpus ${NUM_GPUS}"
fi

# Run pipeline
manifold-genetics pipeline \
    --fit-plink ${FIT_PLINK} \
    --project-plink ${PROJECT_PLINK} \
    --fit-labels ${FIT_LABELS} \
    --project-labels ${PROJECT_LABELS} \
    --colormap ${COLORMAP} \
    --output ${OUTPUT_DIR} \
    --n-pcs 20 \
    --k-min 2 --k-max 10 \
    --embedding phate --knn 500 --t 50 --n-landmark 10000 --random-landmarking \
    --embedding-input fit \
    --admixture-group-column self_described_ancestry \
    --threads ${THREADS} \
    --neuraladmixture-batch-size 400 \
    ${GPU_ARGS} \
    --skip-metrics

echo ""
echo "================================================="
print_success "Pipeline analysis complete!"
echo "================================================="
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
echo "  📊 PCA (fit/project workflow):"
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
echo "Fit/project workflow:"
echo "  • 10K White British samples used for training"
echo "  • 5K Irish samples projected into trained space"
echo ""
