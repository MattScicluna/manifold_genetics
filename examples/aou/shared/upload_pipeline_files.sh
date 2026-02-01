#!/bin/bash
# Upload AoU pipeline files to Google Cloud Storage
# Run from: examples/aou/shared/
#
# Usage:
#   ./upload_pipeline_files.sh                    # Upload both experiments
#   ./upload_pipeline_files.sh 60k_white          # Upload only 60k_white
#   ./upload_pipeline_files.sh hgdp_1kgp_proj     # Upload only hgdp_1kgp_proj

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

# gsutil options for faster uploads:
# -m: parallel uploads
# -o: enable parallel composite uploads for large files (>150MB)
GSUTIL_OPTS="-m -o GSUtil:parallel_composite_upload_threshold=150M"

upload_experiment() {
    local exp_name="$1"
    local exp_dir="$AOU_DIR/$exp_name"

    if [ ! -d "$exp_dir" ]; then
        echo "Error: Experiment directory not found: $exp_dir"
        return 1
    fi

    echo "========================================"
    echo "Uploading: $exp_name"
    echo "========================================"

    # Collect all files to upload
    local files_to_upload=()

    # Data files
    for file in fit_labels.csv project_labels.csv; do
        if [ -f "$exp_dir/data/$file" ]; then
            files_to_upload+=("$exp_dir/data/$file")
        fi
    done

    # Plink files (fit_subset and project_subset)
    for prefix in fit_subset project_subset; do
        for ext in bed bim fam; do
            if [ -f "$exp_dir/data/${prefix}.${ext}" ]; then
                files_to_upload+=("$exp_dir/data/${prefix}.${ext}")
            fi
        done
    done

    # PCA outputs
    for file in fit_pca_20.csv transform_pca_20.csv; do
        if [ -f "$exp_dir/outputs/pca/$file" ]; then
            files_to_upload+=("$exp_dir/outputs/pca/$file")
        fi
    done

    # Admixture outputs
    for k in 2 3 4 5 6 7 8 9 10; do
        for prefix in fit transform; do
            if [ -f "$exp_dir/outputs/admixture/${prefix}.${k}.csv" ]; then
                files_to_upload+=("$exp_dir/outputs/admixture/${prefix}.${k}.csv")
            fi
        done
    done

    echo "Found ${#files_to_upload[@]} files to upload"
    echo "Using parallel uploads with composite upload threshold=150M"
    echo ""

    # Upload all files in parallel, preserving directory structure
    if [ ${#files_to_upload[@]} -gt 0 ]; then
        # Use rsync for efficient parallel upload with directory structure preservation
        # Upload data/ directory
        if [ -d "$exp_dir/data" ]; then
            echo "Uploading data/ directory..."
            gsutil $GSUTIL_OPTS rsync -r "$exp_dir/data/" "$GCS_BASE/$exp_name/data/"
        fi

        # Upload outputs/pca/ directory
        if [ -d "$exp_dir/outputs/pca" ]; then
            echo "Uploading outputs/pca/ directory..."
            gsutil $GSUTIL_OPTS rsync -r "$exp_dir/outputs/pca/" "$GCS_BASE/$exp_name/outputs/pca/"
        fi

        # Upload outputs/admixture/ directory (excluding checkpoints)
        if [ -d "$exp_dir/outputs/admixture" ]; then
            echo "Uploading outputs/admixture/ directory..."
            gsutil $GSUTIL_OPTS rsync -r -x "checkpoints/.*" "$exp_dir/outputs/admixture/" "$GCS_BASE/$exp_name/outputs/admixture/"
        fi
    fi

    echo ""
    echo "Done uploading $exp_name"
    echo ""
}

# Parse arguments
EXPERIMENTS=("60k_white" "hgdp_1kgp_proj")
if [ $# -gt 0 ]; then
    EXPERIMENTS=("$@")
fi

echo "AoU Pipeline Files Upload Script"
echo "================================="
echo "GCS destination: $GCS_BASE"
echo "Experiments: ${EXPERIMENTS[*]}"
echo "Using parallel uploads (-m) with composite threshold=150M"
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    upload_experiment "$exp"
done

echo "========================================"
echo "All uploads complete!"
echo "========================================"
