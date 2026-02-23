#!/bin/bash
# Copy AoU pipeline outputs from manifold_genetics to manifoldGenetics
# Run from: examples/aou/shared/ on AoU
#
# This script copies pre-computed pipeline files to the locations expected
# by manifoldGenetics config files.
#
# Source: manifold_genetics/examples/aou/{60k_white,hgdp_1kgp_proj}/
# Dest:   manifoldGenetics/data/aou/{60K_white,hgdp_1kgp_proj}/
#
# Usage:
#   ./copy_to_manifoldGenetics.sh                    # Copy both experiments
#   ./copy_to_manifoldGenetics.sh 60k_white          # Copy only 60k_white
#   ./copy_to_manifoldGenetics.sh hgdp_1kgp_proj     # Copy only hgdp_1kgp_proj

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_AOU_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COLORMAP_DIR="$(cd "$SCRIPT_DIR/../../colormaps" && pwd)"

# Find manifoldGenetics directory (sibling to manifold_genetics)
if [ -z "${DEST_BASE:-}" ]; then
    MANIFOLD_GENETICS_ROOT="$(cd "$SOURCE_AOU_DIR/../../.." && pwd)"
    if [ -d "$MANIFOLD_GENETICS_ROOT/manifoldGenetics" ]; then
        DEST_BASE="$MANIFOLD_GENETICS_ROOT/manifoldGenetics/data/aou"
    else
        # Try common AoU workspace layout
        WORKSPACE_ROOT="${HOME}/workspaces/phaterepresentationsforvisualizationofgeneticdata"
        if [ -d "$WORKSPACE_ROOT/manifoldGenetics" ]; then
            DEST_BASE="$WORKSPACE_ROOT/manifoldGenetics/data/aou"
        else
            echo "Error: Cannot find manifoldGenetics directory"
            echo "Searched:"
            echo "  - $MANIFOLD_GENETICS_ROOT/manifoldGenetics"
            echo "  - $WORKSPACE_ROOT/manifoldGenetics"
            echo ""
            echo "Set DEST_BASE manually: DEST_BASE=/path/to/manifoldGenetics/data/aou ./copy_to_manifoldGenetics.sh"
            exit 1
        fi
    fi
fi
# Mapping: source directory name -> destination directory name
# Note: manifoldGenetics configs expect 60K_white (uppercase K)
declare -A DIR_MAPPING=(
    ["60k_white"]="60K_white"
    ["10k_WBH"]="10K_WBH"
    ["hgdp_1kgp_proj"]="hgdp_1kgp_proj"
)

copy_experiment() {
    local src_name="$1"
    local dest_name="${DIR_MAPPING[$src_name]:-$src_name}"
    local src_dir="$SOURCE_AOU_DIR/$src_name"
    local dest_dir="$DEST_BASE/$dest_name"

    if [ ! -d "$src_dir" ]; then
        echo "Warning: Source directory not found: $src_dir"
        return 1
    fi

    echo "========================================"
    echo "Copying: $src_name -> $dest_name"
    echo "  From: $src_dir"
    echo "  To:   $dest_dir"
    echo "========================================"

    # Create destination directory structure
    mkdir -p "$dest_dir/data"
    mkdir -p "$dest_dir/outputs/pca"
    mkdir -p "$dest_dir/outputs/admixture"

    # Copy label files
    echo "Copying label files..."
    for file in fit_labels.csv project_labels.csv fit_samples.txt; do
        if [ -f "$src_dir/data/$file" ]; then
            cp -v "$src_dir/data/$file" "$dest_dir/data/"
        fi
    done

    # Copy PCA outputs
    echo "Copying PCA outputs..."
    for file in fit_pca_20.csv transform_pca_20.csv; do
        if [ -f "$src_dir/outputs/pca/$file" ]; then
            cp -v "$src_dir/outputs/pca/$file" "$dest_dir/outputs/pca/"
        fi
    done

    # Copy admixture outputs
    echo "Copying admixture outputs..."
    for k in 2 3 4 5 6 7 8 9 10; do
        for prefix in fit transform; do
            if [ -f "$src_dir/outputs/admixture/${prefix}.${k}.csv" ]; then
                cp -v "$src_dir/outputs/admixture/${prefix}.${k}.csv" "$dest_dir/outputs/admixture/"
            fi
        done
    done

    # Copy colormaps from examples/colormaps/
    echo "Copying colormaps..."
    # AoU colormaps (aou.json used for test set and 60k_white)
    for cmap in aou.json aou_original.json; do
        if [ -f "$COLORMAP_DIR/$cmap" ]; then
            cp -v "$COLORMAP_DIR/$cmap" "$dest_dir/data/"
        fi
    done
    # HGDP+1KGP aligned colormaps (used for train set in hgdp_1kgp_proj)
    if [ "$src_name" = "hgdp_1kgp_proj" ]; then
        for cmap in hgdp_1kgp_aou_aligned.json hgdp_1kgp_aou_aligned_original.json; do
            if [ -f "$COLORMAP_DIR/$cmap" ]; then
                cp -v "$COLORMAP_DIR/$cmap" "$dest_dir/data/"
            fi
        done
    fi

    echo ""
    echo "Done copying $src_name"
    echo ""
}

# Parse arguments
EXPERIMENTS=("60k_white" "10k_WBH" "hgdp_1kgp_proj")
if [ $# -gt 0 ]; then
    EXPERIMENTS=("$@")
fi

echo "Copy AoU Pipeline Files to manifoldGenetics"
echo "============================================"
echo "Source base: $SOURCE_AOU_DIR"
echo "Dest base:   $DEST_BASE"
echo "Experiments: ${EXPERIMENTS[*]}"
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    copy_experiment "$exp"
done

echo "========================================"
echo "All copies complete!"
echo "========================================"
echo ""
echo "Files are now available at:"
for exp in "${EXPERIMENTS[@]}"; do
    dest_name="${DIR_MAPPING[$exp]:-$exp}"
    echo "  $DEST_BASE/$dest_name/"
done
