#!/bin/bash
#
# Run UKBB-HGDP cross-projection analysis pipeline
#
# This script:
# 1. Checks for processed data (runs prepare_data.sh if needed)
# 2. Runs the full cross-projection analysis pipeline
# 3. Generates separate visualizations for HGDP (fit) and UKBB (project) populations
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
echo "This will run the complete cross-projection analysis:"
echo "  - HGDP → UKBB projection (HGDP as reference)"
echo "  - PCA (20 components)"
echo "  - Neural Admixture (K=2-5)"
echo "  - PHATE Embedding"
echo "  - Separate visualizations for HGDP and UKBB populations"
echo ""

# Paths
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
FIT_PLINK="${DATA_DIR}/fit_subset"
PROJECT_PLINK="${DATA_DIR}/project_subset"
HGDP_LABELS="${DATA_DIR}/hgdp_labels.csv"
UKBB_LABELS="${DATA_DIR}/ukbb_labels.csv"
HGDP_COLORMAP="${DATA_DIR}/hgdp_colormap.json"
UKBB_COLORMAP="${DATA_DIR}/ukbb_colormap.json"
THREADS="${SLURM_CPUS_PER_TASK:-4}"
NUM_GPUS="${SLURM_GPUS_ON_NODE:-}"

# Step 1: Check if processed data exists
print_status "Checking for processed data..."

REQUIRED_FILES=(
    "${FIT_PLINK}.bed"
    "${FIT_PLINK}.bim"
    "${FIT_PLINK}.fam"
    "${PROJECT_PLINK}.bed"
    "${PROJECT_PLINK}.bim"
    "${PROJECT_PLINK}.fam"
    "${HGDP_LABELS}"
    "${UKBB_LABELS}"
    "${HGDP_COLORMAP}"
    "${UKBB_COLORMAP}"
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
    echo "Running prepare_data.sh..."
    bash "${SCRIPT_DIR}/prepare_data.sh"
else
    print_success "Processed data found"
fi

# Step 2: Verify all required files exist after preparation
print_status "Verifying required files..."

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        MISSING_FILES+=("$file")
    fi
done

if [[ ${#MISSING_FILES[@]} -gt 0 ]]; then
    echo "Missing required files after data preparation:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Please check the data preparation step and ensure all required files are generated."
    exit 1
fi

print_success "All required files found"

# Step 3: Check if virtual environment is activated
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

echo "  Fit dataset (HGDP): $FIT_SAMPLES samples"
echo "  Project dataset (UKBB): $PROJECT_SAMPLES samples"
echo "  Common SNPs: $SNP_COUNT SNPs"

# Step 5: Run the cross-projection pipeline
print_status "Running cross-projection analysis pipeline..."

echo ""
echo "Command:"
echo "manifold-genetics pipeline \\"
echo "    --fit-plink ${FIT_PLINK} \\"
echo "    --project-plink ${PROJECT_PLINK} \\"
echo "    --labels ${HGDP_LABELS} \\"
echo "    --colormap ${HGDP_COLORMAP} \\"
echo "    --fit-labels ${HGDP_LABELS} \\"
echo "    --project-labels ${UKBB_LABELS} \\"
echo "    --fit-colormap ${HGDP_COLORMAP} \\"
echo "    --project-colormap ${UKBB_COLORMAP} \\"
echo "    --output ${OUTPUT_DIR} \\"
echo "    --n-pcs 20 \\"
echo "    --k-min 2 --k-max 5 \\"
echo "    --embedding phate --knn 100 --t 3 \\"
echo "    --skip-metrics \\"
echo "    --threads ${THREADS}"
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
    --project-labels "$UKBB_LABELS" \
    --fit-colormap "$HGDP_COLORMAP" \
    --project-colormap "$UKBB_COLORMAP" \
    --output "$OUTPUT_DIR" \
    --n-pcs 20 \
    --k-min 2 --k-max 5 \
    --embedding phate --knn 100 --t 3 \
    --skip-metrics \
    --threads "$THREADS" \
    ${NUM_GPUS:+--num-gpus "$NUM_GPUS"}

# Step 6: Summary
echo ""
echo "================================================="
print_success "Cross-projection pipeline analysis complete!"
echo "================================================="
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
echo "  📊 PCA (HGDP fit → UKBB project workflow):"
echo "    - pca/fit_pca_50.csv        (HGDP coordinates)"
echo "    - pca/transform_pca_50.csv  (UKBB coordinates)"
echo "    - figures/pca/pca_pairs_by_*.png (UKBB ancestry plots)"
echo ""
echo "  🧬 Admixture (HGDP fit → UKBB project workflow):"
echo "    - admixture/fit.2.csv ... fit.5.csv        (HGDP Q proportions)"
echo "    - admixture/transform.2.csv ... transform.5.csv (UKBB Q proportions)"
echo "    - figures/admixture/transform_bars.png     (UKBB ancestry bars)"
echo "    - figures/admixture/transform_admixture_colored_embedding.png"
echo ""
echo "  🗺️ PHATE Embedding (on UKBB PCA coordinates):"
echo "    - embeddings/phate_2d.csv"
echo "    - figures/embeddings/phate_by_*.png        (UKBB ancestry plots)"
echo ""
echo "  📈 Metrics:"
echo "    - metrics/geographic_preservation.json     (if geographic data available)"
echo "    - metrics/admixture_preservation.json"
echo ""
echo "Key insights from this cross-projection analysis:"
echo "  ✨ HGDP populations used as diverse reference for model training"
echo "  ✨ UKBB samples projected into HGDP-trained PCA/admixture space"
echo "  ✨ Separate visualizations show population structure for each cohort"
echo "  ✨ Metrics evaluate how well population structure is preserved in projections"
echo ""