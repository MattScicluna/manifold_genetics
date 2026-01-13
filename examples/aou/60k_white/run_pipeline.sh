#!/bin/bash
#
# AoU 60K White Subset Pipeline
#
# This is a minimal wrapper that sets AoU-specific paths and calls
# the generic subsample template.
#
# Usage:
#   bash examples/aou/60k_white/run_pipeline.sh
#

set -e

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ============================================================================
# AoU-Specific Configuration
# ============================================================================

# IMPORTANT: Update these paths to point to your actual data location
# These are placeholder paths - replace with your real data paths before running

# For development: Uncomment and set your actual paths here
# export DATA_DIR="/path/to/your/aou/data"
# export FIT_PLINK="/path/to/aou_60k_white_fit"
# export PROJECT_PLINK="/path/to/aou_60k_white_project"
# export FIT_LABELS="/path/to/fit_labels.csv"
# export PROJECT_LABELS="/path/to/project_labels.csv"
# export COLORMAP="${PROJECT_ROOT}/examples/colormaps/aou.json"
# export OUTPUT_DIR="/path/to/output"

# Default paths (relative to script directory) - will work if data is in ./data/
export DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
export FIT_PLINK="${FIT_PLINK:-${DATA_DIR}/fit_subset}"
export PROJECT_PLINK="${PROJECT_PLINK:-${DATA_DIR}/project_subset}"
export FIT_LABELS="${FIT_LABELS:-${DATA_DIR}/fit_labels.csv}"
export PROJECT_LABELS="${PROJECT_LABELS:-${DATA_DIR}/project_labels.csv}"
export COLORMAP="${COLORMAP:-${PROJECT_ROOT}/examples/colormaps/aou.json}"

# Set pipeline parameters
export N_PCS=20
export K_MIN=2
export K_MAX=10
export EMBEDDING="phate"
export ADMIXTURE_GROUP_COLUMN="race_ethnicity"

# ============================================================================
# Call Generic Subsample Template
# ============================================================================

# Note: Subsample mode uses random landmarking by default
# Note: "$@" passes through any additional arguments (e.g., --skip-metrics)
bash "${PROJECT_ROOT}/examples/generic/subset/run_pipeline.sh" "$@"
