#!/bin/bash
#
# Run complete HGDP+1KGP analysis pipeline
#
# This script:
# 1. Downloads data if needed
# 2. Processes data if needed
# 3. Runs the full analysis pipeline with PCA and embedding visualization
#
# Usage:
#   bash examples/hgdp_1kgp/run_pipeline.sh
#

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

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
echo "  HGDP+1KGP Pipeline Analysis"
echo "========================================="
echo ""
echo "This will run the complete analysis pipeline:"
echo "  - PCA (50 components)"
echo "  - Neural Admixture (K=2-5)"
echo "  - PHATE Embedding"
echo "  - PCA Visualization"
echo "  - Embedding Visualization"
echo ""

# Paths
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
FIT_PLINK="${DATA_DIR}/fit_subset"
PROJECT_PLINK="${DATA_DIR}/project_subset"
LABELS_CSV="${DATA_DIR}/hgdp_project_labels.csv"
COLORMAP_JSON="${DATA_DIR}/colormap.json"
GEOGRAPHIC_CSV="${DATA_DIR}/hgdp_project_geographic.csv"
THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Step 1: Check if raw data exists
print_status "Checking for raw data..."

if [[ ! -f "${DATA_DIR}/raw/full_dataset.bed" ]]; then
    print_warning "Raw data not found, downloading..."
    bash "${SCRIPT_DIR}/download_data.sh"
else
    print_success "Raw data found"
fi

# Step 2: Check if processed data exists
print_status "Checking for processed data..."

if [[ ! -f "${TRANSFORM_PLINK}.bed" ]] || [[ ! -f "${LABELS_CSV}" ]]; then
    print_warning "Processed data not found, creating subsets..."
    bash "${SCRIPT_DIR}/prepare_data.sh"
else
    print_success "Processed data found"
fi

# Step 3: Verify all required files exist
print_status "Verifying required files..."

REQUIRED_FILES=(
    "${FIT_PLINK}.bed"
    "${FIT_PLINK}.bim"
    "${FIT_PLINK}.fam"
    "${PROJECT_PLINK}.bed"
    "${PROJECT_PLINK}.bim"
    "${PROJECT_PLINK}.fam"
    "${LABELS_CSV}"
    "${COLORMAP_JSON}"
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
    echo "Please run the data download and preparation scripts:"
    echo "  bash examples/hgdp_1kgp/download_data.sh"
    echo "  bash examples/hgdp_1kgp/prepare_data.sh"
    exit 1
fi

print_success "All required files found"

# Step 4: Check if virtual environment is activated
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

# Step 5: Run the pipeline
print_status "Running complete analysis pipeline..."

echo ""
echo "Command:"
echo "manifold-genetics pipeline \\"
echo "    --fit-plink ${FIT_PLINK} \\"
echo "    --project-plink ${PROJECT_PLINK} \\"
echo "    --labels ${LABELS_CSV} \\"
echo "    --colormap ${COLORMAP_JSON} \\"
echo "    --geographic ${GEOGRAPHIC_CSV} \\"
echo "    --output ${OUTPUT_DIR} \\"
echo "    --n-pcs 50 \\"
echo "    --k-min 2 --k-max 5 \\"
echo "    --embedding phate --knn 100 --t 3 \\"
echo "    --threads ${THREADS}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run pipeline
manifold-genetics pipeline \
    --fit-plink "$FIT_PLINK" \
    --project-plink "$PROJECT_PLINK" \
    --labels "$LABELS_CSV" \
    --colormap "$COLORMAP_JSON" \
    --geographic "$GEOGRAPHIC_CSV" \
    --output "$OUTPUT_DIR" \
    --n-pcs 50 \
    --k-min 2 --k-max 5 \
    --embedding phate --knn 100 --t 3 \
    --threads "$THREADS"

# Step 6: Summary
echo ""
echo "========================================="
print_success "Pipeline analysis complete!"
echo "========================================="
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
echo "  📊 PCA (fit/transform workflow):"
echo "    - pca/fit_pca_50.csv"
echo "    - pca/transform_pca_50.csv"
echo "    - figures/pca/pca_pairs_by_Population.png"
echo "    - figures/pca/pca_pairs_by_Genetic_region_merged.png"
echo ""
echo "  🧬 Admixture (fit/transform workflow):"
echo "    - admixture/fit_K2.Q ... fit_K5.Q"
echo "    - admixture/transform_K2.Q ... transform_K5.Q"
echo ""
echo "  🗺️ PHATE Embedding (on transform PCA):"
echo "    - embeddings/phate_2d.csv"
echo "    - figures/embeddings/phate_by_Population.png"
echo "    - figures/embeddings/phate_by_Genetic_region_merged.png"
echo ""
echo "  📈 Metrics:"
echo "    - metrics/geographic_preservation.json"
echo "    - metrics/admixture_preservation.json"
echo ""
echo "View the plots to see population structure in PCA and PHATE space!"
echo ""
