#!/bin/bash
#
# Run complete HGDP+1KGP analysis pipeline
#
# This script:
# 1. Downloads data if needed
# 2. Processes data if needed
# 3. Runs the full analysis pipeline using the shared runner
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
echo "  - Neural Admixture (K=2-10)"
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
COLORMAP_JSON="${PROJECT_ROOT}/examples/colormaps/hgdp_1kgp.json"
GEOGRAPHIC_CSV="${DATA_DIR}/hgdp_project_geographic.csv"

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

if [[ ! -f "${PROJECT_PLINK}.bed" ]] || [[ ! -f "${LABELS_CSV}" ]]; then
    print_warning "Processed data not found, creating subsets..."
    bash "${SCRIPT_DIR}/prepare_data.sh"
else
    print_success "Processed data found"
fi

# Step 3: Detect cluster environment
source "${PROJECT_ROOT}/examples/_shared/detect_cluster.sh"

# Step 4: Run shared pipeline
# Note: "$@" passes through any additional arguments (e.g., --skip-admixture for testing)
# Using "transform" mode: fit+transform PHATE on the 4094 transform set (which includes the fit set)
bash "${PROJECT_ROOT}/examples/_shared/run_pipeline.sh" \
    --mode transform \
    --fit-plink "$FIT_PLINK" \
    --project-plink "$PROJECT_PLINK" \
    --labels "$LABELS_CSV" \
    --colormap "$COLORMAP_JSON" \
    --geographic "$GEOGRAPHIC_CSV" \
    --output "$OUTPUT_DIR" \
    --n-pcs 50 \
    --k-min 2 --k-max 10 \
    --embedding phate \
    --admixture-group-column Genetic_region_merged \
    --threads "$CLUSTER_CPUS" \
    ${CLUSTER_GPUS:+--num-gpus "$CLUSTER_GPUS"} \
    "$@"

# Summary
echo ""
echo "Generated files:"
echo "  📊 PCA (fit/transform workflow):"
echo "    - pca/fit_pca_50.csv"
echo "    - pca/project_pca_50.csv"
echo "    - figures/pca/pca_pairs_by_Population.png"
echo "    - figures/pca/pca_pairs_by_Genetic_region_merged.png"
echo ""
echo "  🧬 Admixture (fit/transform workflow):"
echo "    - admixture/fit.2.csv ... fit.10.csv"
echo "    - admixture/project.2.csv ... project.10.csv"
echo "    - figures/admixture/project_bars.png"
echo "    - figures/admixture/project_admixture_colored_embedding.png"
echo ""
echo "  🗺️ PHATE Embedding (on project PCA):"
echo "    - embeddings/phate_2d.csv"
echo "    - figures/embeddings/phate_by_Population.png"
echo "    - figures/embeddings/phate_by_Genetic_region_merged.png"
echo ""
echo "  📈 Metrics:"
echo "    - metrics/geographic.json"
echo "    - metrics/admixture.json"
echo ""
echo "View the plots to see population structure in PCA and PHATE space!"
echo ""
