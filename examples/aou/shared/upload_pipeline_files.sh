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

    # Data files
    echo "Uploading data files..."
    for file in fit_labels.csv project_labels.csv; do
        if [ -f "$exp_dir/data/$file" ]; then
            gsutil cp "$exp_dir/data/$file" "$GCS_BASE/$exp_name/data/$file"
            echo "  Uploaded: data/$file"
        fi
    done

    # Plink files (fit_subset and project_subset)
    echo "Uploading plink files..."
    for prefix in fit_subset project_subset; do
        for ext in bed bim fam; do
            if [ -f "$exp_dir/data/${prefix}.${ext}" ]; then
                gsutil cp "$exp_dir/data/${prefix}.${ext}" "$GCS_BASE/$exp_name/data/${prefix}.${ext}"
                echo "  Uploaded: data/${prefix}.${ext}"
            fi
        done
    done

    # PCA outputs
    echo "Uploading PCA outputs..."
    for file in fit_pca_20.csv transform_pca_20.csv; do
        if [ -f "$exp_dir/outputs/pca/$file" ]; then
            gsutil cp "$exp_dir/outputs/pca/$file" "$GCS_BASE/$exp_name/outputs/pca/$file"
            echo "  Uploaded: outputs/pca/$file"
        fi
    done

    # Admixture outputs
    echo "Uploading admixture outputs..."
    for k in 2 3 4 5 6 7 8 9 10; do
        for prefix in fit transform; do
            if [ -f "$exp_dir/outputs/admixture/${prefix}.${k}.csv" ]; then
                gsutil cp "$exp_dir/outputs/admixture/${prefix}.${k}.csv" "$GCS_BASE/$exp_name/outputs/admixture/${prefix}.${k}.csv"
                echo "  Uploaded: outputs/admixture/${prefix}.${k}.csv"
            fi
        done
    done

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
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    upload_experiment "$exp"
done

echo "========================================"
echo "All uploads complete!"
echo "========================================"
