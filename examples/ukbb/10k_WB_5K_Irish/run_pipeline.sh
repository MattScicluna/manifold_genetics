#!/bin/bash
#
# UKBB 10k WB + 5k Irish Subset Pipeline
#
# This script configures and runs the pipeline on a subsample of the
# UK Biobank (UKBB) cohort.
#
# This script is a minimal wrapper that calls the shared pipeline runner
# with the appropriate mode.
#
# Usage:
#   bash examples/ukbb/10k_WB_5k_Irish/run_pipeline.sh
#

set -e

# ============================================================================ 
# CONFIGURATION
# ============================================================================ 

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# --- IMPORTANT: UPDATE THESE PATHS ---
# Default paths (relative to script directory)
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
FIT_PLINK="${FIT_PLINK:-${DATA_DIR}/fit_subset}"
PROJECT_PLINK="${PROJECT_PLINK:-${DATA_DIR}/project_subset}"
FIT_LABELS="${FIT_LABELS:-${DATA_DIR}/fit_labels.csv}"
PROJECT_LABELS="${PROJECT_LABELS:-${DATA_DIR}/project_labels.csv}"
COLORMAP="${COLORMAP:-${PROJECT_ROOT}/examples/colormaps/ukbb.json}"
# ------------------------------------ 

# ============================================================================ 
# Detect cluster environment
# ============================================================================ 

source "${PROJECT_ROOT}/examples/_shared/detect_cluster.sh"

# ============================================================================ 
# Run pipeline with subsample mode
# ============================================================================ 

# This pipeline uses "subsample" mode, which is designed for visualizing
# large, standalone cohorts.
#
# The shared runner script will automatically apply the correct defaults for this mode:
# - PHATE: knn=500, t=50
# - Random landmarking with 10,000 landmarks
#
bash "${PROJECT_ROOT}/examples/_shared/run_pipeline.sh" \
    --mode subsample \
    --fit-plink "$FIT_PLINK" \
    --project-plink "$PROJECT_PLINK" \
    --fit-labels "$FIT_LABELS" \
    --project-labels "$PROJECT_LABELS" \
    --colormap "$COLORMAP" \
    --output "$OUTPUT_DIR" \
    --n-pcs 20 \
    --k-min 2 \
    --k-max 10 \
    --embedding "phate" \
    --admixture-group-column "self_described_ancestry" \
    --threads "$CLUSTER_CPUS" \
    ${CLUSTER_GPUS:+"--num-gpus" "$CLUSTER_GPUS"} \
    "$@"

echo ""
echo "Pipeline complete! Results in: ${OUTPUT_DIR}"
echo ""