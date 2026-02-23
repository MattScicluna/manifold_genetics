#!/bin/bash
# Download AoU pipeline files from Google Cloud Storage
# Run from: examples/aou/shared/
#
# This script downloads pre-computed pipeline files and populates them in the
# same locations as if the experiments were run locally.
#
# Usage:
#   ./download_pipeline_files.sh                    # Download both experiments
#   ./download_pipeline_files.sh 60k_white          # Download only 60k_white
#   ./download_pipeline_files.sh hgdp_1kgp_proj     # Download only hgdp_1kgp_proj

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AOU_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check WORKSPACE_BUCKET is set
if [ -z "$WORKSPACE_BUCKET" ]; then
    echo "Error: WORKSPACE_BUCKET environment variable is not set"
    echo "Set it with: export WORKSPACE_BUCKET=gs://your-bucket-name"
    exit 1
fi

GCS_BASE="${WORKSPACE_BUCKET}/manifold_genetics/examples/aou"

# gsutil options for faster downloads:
# -m: parallel downloads
GSUTIL_OPTS="-m"

download_experiment() {
    local exp_name="$1"
    local exp_dir="$AOU_DIR/$exp_name"

    echo "========================================"
    echo "Downloading: $exp_name"
    echo "========================================"

    # Create directory structure
    mkdir -p "$exp_dir/data"
    mkdir -p "$exp_dir/outputs/pca"
    mkdir -p "$exp_dir/outputs/admixture"

    echo "Using parallel downloads (-m)"
    echo ""

    # Download data/ directory
    echo "Downloading data/ directory..."
    if gsutil -q stat "$GCS_BASE/$exp_name/data/*" 2>/dev/null; then
        gsutil $GSUTIL_OPTS rsync -r "$GCS_BASE/$exp_name/data/" "$exp_dir/data/"
    else
        echo "  No data files found in GCS"
    fi

    # Download outputs/pca/ directory
    echo "Downloading outputs/pca/ directory..."
    if gsutil -q stat "$GCS_BASE/$exp_name/outputs/pca/*" 2>/dev/null; then
        gsutil $GSUTIL_OPTS rsync -r "$GCS_BASE/$exp_name/outputs/pca/" "$exp_dir/outputs/pca/"
    else
        echo "  No PCA files found in GCS"
    fi

    # Download outputs/admixture/ directory
    echo "Downloading outputs/admixture/ directory..."
    if gsutil -q stat "$GCS_BASE/$exp_name/outputs/admixture/*" 2>/dev/null; then
        gsutil $GSUTIL_OPTS rsync -r "$GCS_BASE/$exp_name/outputs/admixture/" "$exp_dir/outputs/admixture/"
    else
        echo "  No admixture files found in GCS"
    fi

    echo ""
    echo "Done downloading $exp_name"
    echo ""
}

# Parse arguments
EXPERIMENTS=("60k_white" "10k_WBH" "hgdp_1kgp_proj")
if [ $# -gt 0 ]; then
    EXPERIMENTS=("$@")
fi

echo "AoU Pipeline Files Download Script"
echo "==================================="
echo "GCS source: $GCS_BASE"
echo "Experiments: ${EXPERIMENTS[*]}"
echo "Local destination: $AOU_DIR"
echo "Using parallel downloads (-m)"
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    download_experiment "$exp"
done

echo "========================================"
echo "All downloads complete!"
echo "========================================"
echo ""
echo "Files are now available at:"
for exp in "${EXPERIMENTS[@]}"; do
    echo "  $AOU_DIR/$exp/"
done
