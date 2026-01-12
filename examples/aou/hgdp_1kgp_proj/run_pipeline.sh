#!/bin/bash
#
# AoU-HGDP Cross-Projection Pipeline
#
# This is a minimal wrapper that sets AoU-specific paths and calls
# the generic cross-projection template.
#
# Note: Uses landmarking due to large AoU dataset size.
#
# Usage:
#   bash examples/aou/hgdp_1kgp_proj/run_pipeline.sh
#

set -e

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ============================================================================
# AoU-Specific Configuration
# ============================================================================

# Set paths for this dataset
export DATA_DIR="${SCRIPT_DIR}/data"
export OUTPUT_DIR="${SCRIPT_DIR}/outputs"
export FIT_PLINK="${DATA_DIR}/fit_subset"
export PROJECT_PLINK="${DATA_DIR}/project_subset"
export FIT_LABELS="${DATA_DIR}/hgdp_labels.csv"
export PROJECT_LABELS="${DATA_DIR}/aou_labels.csv"
export FIT_COLORMAP="${PROJECT_ROOT}/examples/colormaps/hgdp_1kgp.json"
export PROJECT_COLORMAP="${PROJECT_ROOT}/examples/colormaps/aou.json"

# Set pipeline parameters
export N_PCS=20
export K_MIN=2
export K_MAX=10
export EMBEDDING="phate"
export ADMIXTURE_GROUP_COLUMN="race_ethnicity"
export USE_LANDMARKING=true  # AoU is large, requires landmarking

# ============================================================================
# Call Generic Cross-Projection Template
# ============================================================================

# Note: "$@" passes through any additional arguments (e.g., --skip-metrics)
bash "${PROJECT_ROOT}/examples/generic/hgdp_1kgp_proj/run_pipeline.sh" "$@"
