#!/bin/bash
#
# UKBB-HGDP Cross-Projection Pipeline
#
# This is a minimal wrapper that sets UKBB-specific paths and calls
# the generic cross-projection template.
#
# Usage:
#   bash examples/ukbb/hgdp_1kgp_proj/run_pipeline.sh
#

set -e

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ============================================================================
# UKBB-Specific Configuration
# ============================================================================

# IMPORTANT: Update these paths to point to your actual data location
# These are placeholder paths - replace with your real data paths before running

# For development: Uncomment and set your actual paths here
# export DATA_DIR="/path/to/your/ukbb/data"
# export FIT_PLINK="/path/to/hgdp_1kgp_fit"
# export PROJECT_PLINK="/path/to/ukbb_project"
# export FIT_LABELS="/path/to/hgdp_labels.csv"
# export PROJECT_LABELS="/path/to/ukbb_labels.csv"
# export FIT_COLORMAP="${PROJECT_ROOT}/examples/colormaps/hgdp_1kgp.json"
# export PROJECT_COLORMAP="${PROJECT_ROOT}/examples/colormaps/ukbb.json"
# export OUTPUT_DIR="/path/to/output"

# Default paths (relative to script directory) - will work if data is in ./data/
export DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
export FIT_PLINK="${FIT_PLINK:-${DATA_DIR}/fit_subset}"
export PROJECT_PLINK="${PROJECT_PLINK:-${DATA_DIR}/project_subset}"
export FIT_LABELS="${FIT_LABELS:-${DATA_DIR}/fit_labels.csv}"
export PROJECT_LABELS="${PROJECT_LABELS:-${DATA_DIR}/project_labels.csv}"
export FIT_COLORMAP="${FIT_COLORMAP:-${PROJECT_ROOT}/examples/colormaps/hgdp_1kgp.json}"
export PROJECT_COLORMAP="${PROJECT_COLORMAP:-${PROJECT_ROOT}/examples/colormaps/ukbb.json}"

# Set pipeline parameters
export N_PCS=20
export K_MIN=2
export K_MAX=10
export EMBEDDING="phate"
export ADMIXTURE_GROUP_COLUMN="self_described_ancestry"
export USE_LANDMARKING=false  # UKBB size doesn't require landmarking

# Performance tuning
export NEURALADMIXTURE_BATCH_SIZE=400
export EMBED_BATCH_SIZE=60000

# ============================================================================
# Call Generic Cross-Projection Template
# ============================================================================

# Note: "$@" passes through any additional arguments (e.g., --skip-metrics)
bash "${PROJECT_ROOT}/examples/generic/hgdp_1kgp_proj/run_pipeline.sh" "$@"
