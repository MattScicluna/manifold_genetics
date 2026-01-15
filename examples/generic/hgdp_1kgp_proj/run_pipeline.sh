#!/bin/bash
#
# Generic Cross-Projection Pipeline Template
#
# This template demonstrates how to run cross-cohort projection where you:
# - Fit models on one cohort (reference population)
# - Project/infer on a different cohort (target population)
# - Use separate labels and colormaps for each cohort
#
# This script is a self-contained example. It is not intended to be called
# by other scripts.
#
# Usage:
#   1. Copy this script to your dataset directory
#   2. Update the default paths below to point to your data
#   3. Adjust parameters as needed
#   4. Run: bash run_pipeline.sh
#

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# --- IMPORTANT: UPDATE THESE PATHS ---
# Default paths (relative to script directory)
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
FIT_PLINK="${FIT_PLINK:-${DATA_DIR}/fit_subset}"
PROJECT_PLINK="${PROJECT_PLINK:-${DATA_DIR}/project_subset}"
FIT_LABELS="${FIT_LABELS:-${DATA_DIR}/fit_labels.csv}"
PROJECT_LABels="${PROJECT_LABELS:-${DATA_DIR}/project_labels.csv}"
FIT_COLORMAP="${FIT_COLORMAP:-${PROJECT_ROOT}/examples/colormaps/hgdp_1kgp.json}"
PROJECT_COLORMAP="${PROJECT_COLORMAP:-${PROJECT_ROOT}/examples/colormaps/custom.json}"
# ------------------------------------

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
echo ""

# Build command array
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
    --n-pcs 20
    --k-min 2
    --k-max 10
    --embedding "phate"
    --threads "$CLUSTER_CPUS"
    # --- OPTIONAL: Add performance tuning or other flags here ---
    # --n-landmark 10000
    # --random-landmarking
    # --neuraladmixture-batch-size 400
    # --skip-admixture
)

# Add GPU flag if available
${CLUSTER_GPUS:+--num-gpus "$CLUSTER_GPUS"}

# Execute with any additional arguments passed from the command line
"${CMD[@]}" "$@"

echo ""
echo "Pipeline complete! Results in: ${OUTPUT_DIR}"
echo ""