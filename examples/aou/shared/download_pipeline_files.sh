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

    # Data files
    echo "Downloading data files..."
    for file in fit_labels.csv project_labels.csv; do
        if gsutil -q stat "$GCS_BASE/$exp_name/data/$file" 2>/dev/null; then
            gsutil cp "$GCS_BASE/$exp_name/data/$file" "$exp_dir/data/$file"
            echo "  Downloaded: data/$file"
        else
            echo "  Skipped (not found): data/$file"
        fi
    done

    # Plink files (fit_subset and project_subset)
    echo "Downloading plink files..."
    for prefix in fit_subset project_subset; do
        for ext in bed bim fam; do
            if gsutil -q stat "$GCS_BASE/$exp_name/data/${prefix}.${ext}" 2>/dev/null; then
                gsutil cp "$GCS_BASE/$exp_name/data/${prefix}.${ext}" "$exp_dir/data/${prefix}.${ext}"
                echo "  Downloaded: data/${prefix}.${ext}"
            else
                echo "  Skipped (not found): data/${prefix}.${ext}"
            fi
        done
    done

    # PCA outputs
    echo "Downloading PCA outputs..."
    for file in fit_pca_20.csv transform_pca_20.csv; do
        if gsutil -q stat "$GCS_BASE/$exp_name/outputs/pca/$file" 2>/dev/null; then
            gsutil cp "$GCS_BASE/$exp_name/outputs/pca/$file" "$exp_dir/outputs/pca/$file"
            echo "  Downloaded: outputs/pca/$file"
        else
            echo "  Skipped (not found): outputs/pca/$file"
        fi
    done

    # Admixture outputs
    echo "Downloading admixture outputs..."
    for k in 2 3 4 5 6 7 8 9 10; do
        for prefix in fit transform; do
            if gsutil -q stat "$GCS_BASE/$exp_name/outputs/admixture/${prefix}.${k}.csv" 2>/dev/null; then
                gsutil cp "$GCS_BASE/$exp_name/outputs/admixture/${prefix}.${k}.csv" "$exp_dir/outputs/admixture/${prefix}.${k}.csv"
                echo "  Downloaded: outputs/admixture/${prefix}.${k}.csv"
            else
                echo "  Skipped (not found): outputs/admixture/${prefix}.${k}.csv"
            fi
        done
    done

    echo ""
    echo "Done downloading $exp_name"
    echo ""
}

# Parse arguments
EXPERIMENTS=("60k_white" "hgdp_1kgp_proj")
if [ $# -gt 0 ]; then
    EXPERIMENTS=("$@")
fi

echo "AoU Pipeline Files Download Script"
echo "==================================="
echo "GCS source: $GCS_BASE"
echo "Experiments: ${EXPERIMENTS[*]}"
echo "Local destination: $AOU_DIR"
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
