#!/bin/bash
#
# Generic Subset Pipeline Template
#
# This template demonstrates how to run the pipeline on your own dataset
# using the shared pipeline runner with subsample mode.
#
# Subsample mode is for large datasets where you:
# - Fit models on a random subsample
# - Project/infer on the full dataset
# - Use landmarking for computational efficiency
#
# Usage:
#   1. Copy this script to your dataset directory
#   2. Update the paths below to point to your data
#   3. Adjust parameters as needed
#   4. Run: bash run_pipeline.sh
#

set -e

# ============================================================================
# CONFIGURATION - Update these paths for your dataset
# ============================================================================

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Dataset paths (UPDATE THESE)
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
FIT_PLINK="${DATA_DIR}/fit_subset"          # Random subsample for training
PROJECT_PLINK="${DATA_DIR}/project_subset"  # Full dataset for projection
FIT_LABELS="${DATA_DIR}/fit_labels.csv"
PROJECT_LABELS="${DATA_DIR}/project_labels.csv"
COLORMAP="${PROJECT_ROOT}/examples/colormaps/your_colormap.json"

# Pipeline parameters (ADJUST AS NEEDED)
N_PCS=20                           # Number of PCA components
K_MIN=2                            # Minimum K for admixture
K_MAX=10                           # Maximum K for admixture
EMBEDDING="phate"                  # Embedding method (phate, umap, tsne, diffusion_map)
ADMIXTURE_GROUP_COLUMN=""          # Column for admixture grouping (optional)

# ============================================================================
# Verify files exist
# ============================================================================

echo "Verifying required files..."

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
    echo "ERROR: Missing required files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Please prepare your data first."
    exit 1
fi

echo "All required files found."
echo ""

# ============================================================================
# Detect cluster environment
# ============================================================================

source "${PROJECT_ROOT}/examples/_shared/detect_cluster.sh"

echo "Running on cluster: $CLUSTER_NAME"
echo "CPUs: $CLUSTER_CPUS, GPUs: $CLUSTER_GPUS"
echo ""

# ============================================================================
# Run pipeline with subsample mode
# ============================================================================

# Option 1: Use subsample mode with default parameters
# (knn=500, t=50, n-landmark=10000, random-landmarking, embedding-input=fit)
# Note: "$@" passes through any additional arguments (e.g., --skip-admixture for testing)

bash "${PROJECT_ROOT}/examples/_shared/run_pipeline.sh" \
    --mode subsample \
    --fit-plink "$FIT_PLINK" \
    --project-plink "$PROJECT_PLINK" \
    --fit-labels "$FIT_LABELS" \
    --project-labels "$PROJECT_LABELS" \
    --colormap "$COLORMAP" \
    --output "$OUTPUT_DIR" \
    --n-pcs "$N_PCS" \
    --k-min "$K_MIN" \
    --k-max "$K_MAX" \
    --embedding "$EMBEDDING" \
    ${ADMIXTURE_GROUP_COLUMN:+--admixture-group-column "$ADMIXTURE_GROUP_COLUMN"} \
    --threads "$CLUSTER_CPUS" \
    ${CLUSTER_GPUS:+--num-gpus "$CLUSTER_GPUS"} \
    "$@"

# Option 2: Override specific parameters while using subsample mode
# Uncomment to use custom parameters:
#
# bash "${PROJECT_ROOT}/examples/_shared/run_pipeline.sh" \
#     --mode subsample \
#     --fit-plink "$FIT_PLINK" \
#     --project-plink "$PROJECT_PLINK" \
#     --fit-labels "$FIT_LABELS" \
#     --project-labels "$PROJECT_LABELS" \
#     --colormap "$COLORMAP" \
#     --output "$OUTPUT_DIR" \
#     --n-pcs 50 \
#     --knn 1000 \
#     --t 100 \
#     --n-landmark 5000 \
#     --embedding umap \
#     --threads "$CLUSTER_CPUS"

echo ""
echo "Pipeline complete! Results in: ${OUTPUT_DIR}"
echo ""
