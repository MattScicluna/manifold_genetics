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
# Usage Option 1 (Direct):
#   1. Copy this script to your dataset directory
#   2. Update the default paths below to point to your data
#   3. Adjust parameters as needed
#   4. Run: bash run_pipeline.sh
#
# Usage Option 2 (Called from another script):
#   Set environment variables and call this script:
#     export FIT_PLINK=/path/to/fit
#     export PROJECT_PLINK=/path/to/project
#     export FIT_LABELS=/path/to/fit_labels.csv
#     export PROJECT_LABELS=/path/to/project_labels.csv
#     export FIT_COLORMAP=/path/to/fit_colormap.json
#     export PROJECT_COLORMAP=/path/to/project_colormap.json
#     export OUTPUT_DIR=/path/to/output
#     bash examples/generic/hgdp_1kgp_proj/run_pipeline.sh
#

set -e

# ============================================================================
# CONFIGURATION - Accepts environment variables or uses defaults
# ============================================================================

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Dataset paths - Use environment variables if set, otherwise use defaults
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
FIT_PLINK="${FIT_PLINK:-${DATA_DIR}/fit_subset}"          # Reference cohort (e.g., HGDP)
PROJECT_PLINK="${PROJECT_PLINK:-${DATA_DIR}/project_subset}"  # Target cohort (e.g., UKBB)
FIT_LABELS="${FIT_LABELS:-${DATA_DIR}/fit_labels.csv}"     # Labels for reference cohort
PROJECT_LABELS="${PROJECT_LABELS:-${DATA_DIR}/project_labels.csv}"  # Labels for target cohort
FIT_COLORMAP="${FIT_COLORMAP:-${PROJECT_ROOT}/examples/colormaps/hgdp_1kgp.json}"  # Reference colormap
PROJECT_COLORMAP="${PROJECT_COLORMAP:-${PROJECT_ROOT}/examples/colormaps/ukbb.json}"    # Target colormap

# Pipeline parameters - Use environment variables if set, otherwise use defaults
N_PCS="${N_PCS:-20}"                           # Number of PCA components
K_MIN="${K_MIN:-2}"                            # Minimum K for admixture
K_MAX="${K_MAX:-10}"                           # Maximum K for admixture
EMBEDDING="${EMBEDDING:-phate}"                # Embedding method (phate, umap, tsne, diffusion_map)
ADMIXTURE_GROUP_COLUMN="${ADMIXTURE_GROUP_COLUMN:-}"  # Column for admixture grouping (optional)
USE_LANDMARKING="${USE_LANDMARKING:-false}"    # Set to "true" for large target datasets (AoU)

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

echo ""
echo "Running cross-projection pipeline..."
echo "  Mode: projection (fit on reference, transform on target)"
if [[ "$USE_LANDMARKING" == "true" ]]; then
    echo "  Performance: Large dataset mode (landmarking enabled)"
else
    echo "  Performance: Standard mode (no landmarking)"
fi
echo ""

# Build command with mode and base parameters
CMD=(
    bash "${PROJECT_ROOT}/examples/_shared/run_pipeline.sh"
    --mode projection
    --fit-plink "$FIT_PLINK"
    --project-plink "$PROJECT_PLINK"
    --fit-labels "$FIT_LABELS"
    --project-labels "$PROJECT_LABELS"
    --fit-colormap "$FIT_COLORMAP"
    --project-colormap "$PROJECT_COLORMAP"
    --output "$OUTPUT_DIR"
    --n-pcs "$N_PCS"
    --k-min "$K_MIN"
    --k-max "$K_MAX"
    --embedding "$EMBEDDING"
    --threads "$CLUSTER_CPUS"
)

# Add optional parameters
[[ -n "$ADMIXTURE_GROUP_COLUMN" ]] && CMD+=(--admixture-group-column "$ADMIXTURE_GROUP_COLUMN")
[[ -n "$CLUSTER_GPUS" ]] && CMD+=(--num-gpus "$CLUSTER_GPUS")

# Add landmarking for large datasets (e.g., AoU)
if [[ "$USE_LANDMARKING" == "true" ]]; then
    CMD+=(
        --knn 500
        --t 50
        --n-landmark 10000
        --random-landmarking
        --neuraladmixture-batch-size 400
    )
fi

# Execute with any additional arguments passed through
"${CMD[@]}" "$@"

echo ""
echo "Pipeline complete! Results in: ${OUTPUT_DIR}"
echo ""
