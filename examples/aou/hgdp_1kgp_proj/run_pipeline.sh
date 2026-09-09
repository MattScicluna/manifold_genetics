#!/bin/bash
#
# AoU-HGDP Cross-Projection Pipeline
#
# This script configures and runs the cross-projection pipeline, fitting
# models on HGDP+1KGP and projecting them onto the All of Us (AoU) cohort.
#
# This script is a minimal wrapper that calls the shared pipeline runner
# with the appropriate mode.
#
# Usage:
#   bash examples/aou/hgdp_1kgp_proj/run_pipeline.sh
#

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# --- IMPORTANT: UPDATE THESE PATHS ---
# Default paths (relative to script directory) - will work if data is in ./data/
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
FIT_PLINK="${FIT_PLINK:-${DATA_DIR}/fit_subset}"
PROJECT_PLINK="${PROJECT_PLINK:-${DATA_DIR}/project_subset}"
FIT_LABELS="${FIT_LABELS:-${DATA_DIR}/fit_labels.csv}"
PROJECT_LABELS="${PROJECT_LABELS:-${DATA_DIR}/project_labels.csv}"
FIT_COLORMAP="${FIT_COLORMAP:-${PROJECT_ROOT}/examples/colormaps/hgdp_1kgp_aou_aligned.json}"
PROJECT_COLORMAP="${PROJECT_COLORMAP:-${PROJECT_ROOT}/examples/colormaps/aou.json}"
# ------------------------------------

# ============================================================================
# Detect cluster environment
# ============================================================================

source "${PROJECT_ROOT}/examples/_shared/detect_cluster.sh"

# ============================================================================
# Run pipeline with projection mode
# ============================================================================

# This pipeline uses "projection" mode, which is designed for projecting a
# large cohort onto a smaller reference panel (e.g., AoU onto HGDP+1KGP).
#
# The shared runner script will automatically apply the correct defaults for this mode:
# - PHATE: knn=100, t=3
# - No landmarking (as AoU is the project set, not the fit set for embedding)
#
bash "${PROJECT_ROOT}/examples/_shared/run_pipeline.sh" \
    --mode projection \
    --fit-plink "$FIT_PLINK" \
    --project-plink "$PROJECT_PLINK" \
    --fit-labels "$FIT_LABELS" \
    --project-labels "$PROJECT_LABELS" \
    --fit-colormap "$FIT_COLORMAP" \
    --project-colormap "$PROJECT_COLORMAP" \
    --output "$OUTPUT_DIR" \
    --n-pcs 20 \
    --k-min 2 \
    --k-max 10 \
    --embedding "phate" \
    --admixture-group-column "race" \
    --projection-plot-fit-column "Population" \
    --projection-plot-project-column "race_ethnicity" \
    --threads "$CLUSTER_CPUS" \
    --embed-batch-size 60000 \
    ${CLUSTER_GPUS:+--num-gpus "$CLUSTER_GPUS"} \
    "$@"

echo ""
echo "Pipeline complete! Results in: ${OUTPUT_DIR}"
echo ""