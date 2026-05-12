#!/bin/bash
#
# UKBB Geosketch PHATE Pipeline
#
# Runs PHATE (knn=500, t=50) on the geosketch-selected UKBB fit subset.
# The fit subset was created by prepare_data.sh using geometric sketching to
# select a representative sample from the full intersected UKBB dataset.
#
# Prerequisites:
#   bash examples/ukbb/geosketch_phate/prepare_data.sh
#
# Usage:
#   bash examples/ukbb/geosketch_phate/run_pipeline.sh
#   # or submit as a batch job:
#   bash examples/_shared/submit_batch.sh examples/ukbb/geosketch_phate/run_pipeline.sh
#

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
FIT_PLINK="${FIT_PLINK:-${DATA_DIR}/fit_subset}"
PROJECT_PLINK="${PROJECT_PLINK:-${DATA_DIR}/project_subset}"
FIT_LABELS="${FIT_LABELS:-${DATA_DIR}/fit_labels.csv}"
PROJECT_LABELS="${PROJECT_LABELS:-${DATA_DIR}/project_labels.csv}"
COLORMAP="${COLORMAP:-${PROJECT_ROOT}/examples/colormaps/ukbb.json}"

# ============================================================================
# Detect cluster environment
# ============================================================================

source "${PROJECT_ROOT}/examples/_shared/detect_cluster.sh"

# ============================================================================
# Run pipeline with subsample mode
# ============================================================================
#
# subsample mode defaults: knn=500, t=50, n-landmark=10000, random-landmarking
# These match the knn=500, t=50 used in all large-cohort PHATE runs.
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
    ${CLUSTER_GPUS:+--num-gpus "$CLUSTER_GPUS"} \
    "$@"

echo ""
echo "Pipeline complete! Results in: ${OUTPUT_DIR}"
echo ""
