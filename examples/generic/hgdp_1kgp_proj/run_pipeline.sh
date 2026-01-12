#!/bin/bash
#
# Generic Cross-Projection Pipeline Template
#
# This template demonstrates how to run cross-cohort projection where you:
# - Fit models on one cohort (reference population)
# - Project/infer on a different cohort (target population)
# - Use separate labels and colormaps for each cohort
#
# Example use cases:
# - HGDP+1KGP (reference) → UKBB (target)
# - HGDP+1KGP (reference) → AoU (target)
# - Any diverse reference → specific population target
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
FIT_PLINK="${DATA_DIR}/fit_subset"          # Reference cohort (e.g., HGDP)
PROJECT_PLINK="${DATA_DIR}/project_subset"  # Target cohort (e.g., UKBB)
FIT_LABELS="${DATA_DIR}/fit_labels.csv"     # Labels for reference cohort
PROJECT_LABELS="${DATA_DIR}/project_labels.csv"  # Labels for target cohort
FIT_COLORMAP="${PROJECT_ROOT}/examples/colormaps/hgdp_1kgp.json"  # Reference colormap
PROJECT_COLORMAP="${PROJECT_ROOT}/examples/colormaps/ukbb.json"    # Target colormap

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
    "${FIT_COLORMAP}"
    "${PROJECT_COLORMAP}"
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
# Run pipeline with projection mode
# ============================================================================

# Option 1: Use projection mode with default parameters
# (knn=100, t=3, embedding-input=project)
# Note: "$@" passes through any additional arguments (e.g., --skip-admixture for testing)

bash "${PROJECT_ROOT}/examples/_shared/run_pipeline.sh" \
    --mode projection \
    --fit-plink "$FIT_PLINK" \
    --project-plink "$PROJECT_PLINK" \
    --fit-labels "$FIT_LABELS" \
    --project-labels "$PROJECT_LABELS" \
    --fit-colormap "$FIT_COLORMAP" \
    --project-colormap "$PROJECT_COLORMAP" \
    --output "$OUTPUT_DIR" \
    --n-pcs "$N_PCS" \
    --k-min "$K_MIN" \
    --k-max "$K_MAX" \
    --embedding "$EMBEDDING" \
    ${ADMIXTURE_GROUP_COLUMN:+--admixture-group-column "$ADMIXTURE_GROUP_COLUMN"} \
    --threads "$CLUSTER_CPUS" \
    ${CLUSTER_GPUS:+--num-gpus "$CLUSTER_GPUS"} \
    "$@"

# Option 2: For large target cohorts, use landmarking despite cross-projection
# (e.g., HGDP → large AoU dataset)
# Uncomment to use:
#
# bash "${PROJECT_ROOT}/examples/_shared/run_pipeline.sh" \
#     --fit-plink "$FIT_PLINK" \
#     --project-plink "$PROJECT_PLINK" \
#     --fit-labels "$FIT_LABELS" \
#     --project-labels "$PROJECT_LABELS" \
#     --fit-colormap "$FIT_COLORMAP" \
#     --project-colormap "$PROJECT_COLORMAP" \
#     --output "$OUTPUT_DIR" \
#     --n-pcs "$N_PCS" \
#     --k-min "$K_MIN" \
#     --k-max "$K_MAX" \
#     --embedding "$EMBEDDING" \
#     --knn 500 \
#     --t 50 \
#     --n-landmark 10000 \
#     --random-landmarking \
#     --embedding-input project \
#     --threads "$CLUSTER_CPUS"

echo ""
echo "Pipeline complete! Results in: ${OUTPUT_DIR}"
echo ""
