#!/bin/bash
#
# Redo LD pruning with different parameters
#
# This script allows you to experiment with different LD pruning parameters
# WITHOUT re-running the expensive filtering and standardization steps.
#
# Prerequisites:
#   - Must have run prepare_data.sh at least once to create standardized files
#   - Requires: data/temp/hgdp_standardized.{bed,bim,fam}
#   - Requires: data/temp/aou_standardized.{bed,bim,fam}
#
# Usage:
#   bash redo_ld_pruning.sh                           # Use defaults (50kb, step 10, r²=0.2)
#   LD_R2=0.15 bash redo_ld_pruning.sh                # Stricter (fewer SNPs)
#   LD_R2=0.25 bash redo_ld_pruning.sh                # Looser (more SNPs)
#   LD_WINDOW=150 LD_STEP=1 LD_R2=0.05 bash redo_ld_pruning.sh  # Match R pipeline
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
TEMP_DIR="${DATA_DIR}/temp"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# LD pruning parameters (override via environment variables)
LD_WINDOW="${LD_WINDOW:-50}"    # Window size in kb
LD_STEP="${LD_STEP:-10}"        # Step size
LD_R2="${LD_R2:-0.2}"           # r² threshold (higher = more SNPs retained)

echo "=========================================="
echo "  Redo LD Pruning"
echo "=========================================="
echo ""
echo "Parameters:"
echo "  LD window: ${LD_WINDOW}kb"
echo "  LD step: ${LD_STEP}"
echo "  LD r²: ${LD_R2}"
echo ""

# Check prerequisites
if [[ ! -f "${TEMP_DIR}/hgdp_standardized.bed" ]] || [[ ! -f "${TEMP_DIR}/aou_standardized.bed" ]]; then
    echo "ERROR: Standardized files not found!"
    echo "  Required: ${TEMP_DIR}/hgdp_standardized.{bed,bim,fam}"
    echo "  Required: ${TEMP_DIR}/aou_standardized.{bed,bim,fam}"
    echo ""
    echo "Please run prepare_data.sh first to create these files."
    exit 1
fi

# Find plink2
PLINK2=""
if [[ -f "${PROJECT_ROOT}/bin/plink2" ]]; then
    PLINK2="${PROJECT_ROOT}/bin/plink2"
elif command -v plink2 &> /dev/null; then
    PLINK2="plink2"
else
    echo "ERROR: plink2 not found!"
    exit 1
fi
echo "Using plink2: ${PLINK2}"
echo ""

# Clean up previous LD pruning results
echo "Cleaning up previous LD pruning results..."
rm -f "${TEMP_DIR}/hgdp_prune."*
rm -f "${TEMP_DIR}/hgdp_pruned."*
rm -f "${TEMP_DIR}/aou_pruned."*
rm -f "${TEMP_DIR}/hgdp_snps.txt"
rm -f "${TEMP_DIR}/aou_snps.txt"
rm -f "${TEMP_DIR}/common_snps.txt"
rm -f "${TEMP_DIR}/final_common_snps.txt"
rm -f "${TEMP_DIR}/flip_list.txt"
rm -f "${TEMP_DIR}/exclude_list.txt"
rm -f "${TEMP_DIR}/hgdp_intersected."*
rm -f "${TEMP_DIR}/aou_intersected."*
rm -f "${DATA_DIR}/fit_subset."*
rm -f "${DATA_DIR}/project_subset."*
echo ""

# =============================================================================
# Step 1: LD prune reference and apply to AoU
# =============================================================================
echo "=========================================="
echo "Step 1: LD pruning"
echo "=========================================="

HGDP_PRUNE_PREFIX="${TEMP_DIR}/hgdp_prune"
HGDP_PRUNED_PREFIX="${TEMP_DIR}/hgdp_pruned"
AOU_PRUNED_PREFIX="${TEMP_DIR}/aou_pruned"

echo "  Computing prune list on HGDP reference..."
${PLINK2} --bfile ${TEMP_DIR}/hgdp_standardized \
    --indep-pairwise ${LD_WINDOW} ${LD_STEP} ${LD_R2} \
    --memory 100000 \
    --out ${HGDP_PRUNE_PREFIX}

PRUNE_IN_COUNT=$(wc -l < "${HGDP_PRUNE_PREFIX}.prune.in")
echo "  SNPs retained after LD pruning: ${PRUNE_IN_COUNT}"

echo "  Applying prune list to HGDP reference..."
${PLINK2} --bfile ${TEMP_DIR}/hgdp_standardized \
    --extract ${HGDP_PRUNE_PREFIX}.prune.in \
    --make-bed \
    --out ${HGDP_PRUNED_PREFIX}

echo "  Applying prune list to AoU dataset..."
${PLINK2} --bfile ${TEMP_DIR}/aou_standardized \
    --extract ${HGDP_PRUNE_PREFIX}.prune.in \
    --memory 100000 \
    --make-bed \
    --out ${AOU_PRUNED_PREFIX}

HGDP_PLINK_PREFIX="${HGDP_PRUNED_PREFIX}"
AOU_PLINK_PREFIX="${AOU_PRUNED_PREFIX}"
echo ""

# =============================================================================
# Step 2: Find SNP intersection
# =============================================================================
echo "=========================================="
echo "Step 2: Find SNP intersection"
echo "=========================================="

awk '{print $2}' "${HGDP_PLINK_PREFIX}.bim" | sort > "${TEMP_DIR}/hgdp_snps.txt"
awk '{print $2}' "${AOU_PLINK_PREFIX}.bim" | sort > "${TEMP_DIR}/aou_snps.txt"

HGDP_SNP_COUNT=$(wc -l < "${TEMP_DIR}/hgdp_snps.txt")
AOU_SNP_COUNT=$(wc -l < "${TEMP_DIR}/aou_snps.txt")

echo "  HGDP pruned: $HGDP_SNP_COUNT SNPs"
echo "  AoU pruned: $AOU_SNP_COUNT SNPs"

comm -12 "${TEMP_DIR}/hgdp_snps.txt" "${TEMP_DIR}/aou_snps.txt" > "${TEMP_DIR}/common_snps.txt"

COMMON_SNP_COUNT=$(wc -l < "${TEMP_DIR}/common_snps.txt")
echo "  Common SNPs: $COMMON_SNP_COUNT"

if [[ $COMMON_SNP_COUNT -lt 50000 ]]; then
    echo "  WARNING: Less than 50K common SNPs!"
fi
echo ""

# =============================================================================
# Step 3: Check allele compatibility
# =============================================================================
echo "=========================================="
echo "Step 3: Check allele compatibility"
echo "=========================================="

python3 << EOF
import sys
from pathlib import Path

sys.path.insert(0, str(Path("${PROJECT_ROOT}/src")))
from manifold_genetics.utils.io import check_allele_compatibility

hgdp_bim = Path("${HGDP_PLINK_PREFIX}.bim")
aou_bim = Path("${AOU_PLINK_PREFIX}.bim")
common_snps_file = Path("${TEMP_DIR}/common_snps.txt")
output_dir = Path("${TEMP_DIR}")

with open(common_snps_file, 'r') as f:
    common_snps = {line.strip() for line in f if line.strip()}

exact_matches, need_flip, incompatible = check_allele_compatibility(
    hgdp_bim, aou_bim, common_snps
)

with open(output_dir / "flip_list.txt", 'w') as f:
    for snp in sorted(need_flip):
        f.write(f"{snp}\n")

with open(output_dir / "exclude_list.txt", 'w') as f:
    for snp in sorted(incompatible):
        f.write(f"{snp}\n")
EOF

FLIP_COUNT=$(wc -l < "${TEMP_DIR}/flip_list.txt" 2>/dev/null || echo "0")
EXCLUDE_COUNT=$(wc -l < "${TEMP_DIR}/exclude_list.txt" 2>/dev/null || echo "0")

echo "  SNPs to flip: $FLIP_COUNT"
echo "  SNPs to exclude: $EXCLUDE_COUNT"

if [[ $EXCLUDE_COUNT -gt 0 ]]; then
    comm -23 <(sort "${TEMP_DIR}/common_snps.txt") <(sort "${TEMP_DIR}/exclude_list.txt") > "${TEMP_DIR}/final_common_snps.txt"
else
    cp "${TEMP_DIR}/common_snps.txt" "${TEMP_DIR}/final_common_snps.txt"
fi

FINAL_SNP_COUNT=$(wc -l < "${TEMP_DIR}/final_common_snps.txt")
echo "  Final usable SNPs: $FINAL_SNP_COUNT"
echo ""

# =============================================================================
# Step 4: Create final datasets
# =============================================================================
echo "=========================================="
echo "Step 4: Create final datasets"
echo "=========================================="

echo "  Creating HGDP intersected dataset..."
HGDP_CMD="${PLINK2} --bfile ${HGDP_PLINK_PREFIX} --extract ${TEMP_DIR}/final_common_snps.txt"
if [[ $FLIP_COUNT -gt 0 ]]; then
    HGDP_CMD="$HGDP_CMD --flip ${TEMP_DIR}/flip_list.txt"
fi
HGDP_CMD="$HGDP_CMD --make-bed --out ${TEMP_DIR}/hgdp_intersected"
eval $HGDP_CMD

echo "  Creating AoU intersected dataset..."
${PLINK2} --bfile ${AOU_PLINK_PREFIX} \
    --extract ${TEMP_DIR}/final_common_snps.txt \
    --memory 100000 \
    --make-bed \
    --out ${TEMP_DIR}/aou_intersected

# Copy to final locations
cp "${TEMP_DIR}/hgdp_intersected.bed" "${DATA_DIR}/fit_subset.bed"
cp "${TEMP_DIR}/hgdp_intersected.bim" "${DATA_DIR}/fit_subset.bim"
cp "${TEMP_DIR}/hgdp_intersected.fam" "${DATA_DIR}/fit_subset.fam"

cp "${TEMP_DIR}/aou_intersected.bed" "${DATA_DIR}/project_subset.bed"
cp "${TEMP_DIR}/aou_intersected.bim" "${DATA_DIR}/project_subset.bim"
cp "${TEMP_DIR}/aou_intersected.fam" "${DATA_DIR}/project_subset.fam"

FIT_SAMPLES=$(wc -l < "${DATA_DIR}/fit_subset.fam")
PROJECT_SAMPLES=$(wc -l < "${DATA_DIR}/project_subset.fam")

echo ""
echo "=========================================="
echo "  Complete!"
echo "=========================================="
echo ""
echo "LD pruning parameters used:"
echo "  Window: ${LD_WINDOW}kb, Step: ${LD_STEP}, r²: ${LD_R2}"
echo ""
echo "Results:"
echo "  fit_subset (HGDP): $FIT_SAMPLES samples, $FINAL_SNP_COUNT SNPs"
echo "  project_subset (AoU): $PROJECT_SAMPLES samples, $FINAL_SNP_COUNT SNPs"
echo ""
echo "To try different parameters:"
echo "  LD_R2=0.15 bash redo_ld_pruning.sh   # Stricter (fewer SNPs)"
echo "  LD_R2=0.25 bash redo_ld_pruning.sh   # Looser (more SNPs)"
echo "  LD_WINDOW=150 LD_STEP=1 LD_R2=0.05 bash redo_ld_pruning.sh  # R pipeline style"
echo ""
