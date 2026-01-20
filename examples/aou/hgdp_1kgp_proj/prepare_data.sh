#!/bin/bash
#
# Prepare AoU-HGDP cross-projection data
# This script integrates the aou_pipeline steps to:
# 1. Download/prepare HGDP+1KGP reference data
# 2. Download AoU genotype data (via shared script)
# 3. Filter both datasets (MAF, geno, indel removal)
# 4. LD prune and find common SNPs
# 5. Create fit/project subsets
# 6. Create labels and colormaps
#

set -e

# AOU specific
export SLURM_CPUS_PER_TASK=32

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Configuration
CPU_CORES="${SLURM_CPUS_PER_TASK:-4}"
REF_DATA_URL="gs://fc-secure-47ccf5a8-b9ba-460a-aa03-dea8d260953b/Data/1KGPHGDP.tar.gz"
GOOGLE_PROJECT="${GOOGLE_PROJECT:-}"
CDR_VERSION="${WORKSPACE_CDR:-}"

# =============================================================================
# Filtering parameters (adjust to control final SNP count)
# =============================================================================
# Target: ~150-200K SNPs after all filtering
MAF_THRESHOLD="${MAF_THRESHOLD:-0.01}"    # Minor allele frequency (remove rare variants, 1% default)
GENO_THRESHOLD="${GENO_THRESHOLD:-0.05}"  # Missing genotype rate (remove low-quality SNPs, 5% default)
LD_WINDOW="${LD_WINDOW:-150}"             # LD pruning window size (kb)
LD_STEP="${LD_STEP:-1}"                   # LD pruning step size
LD_R2="${LD_R2:-0.05}"                    # LD pruning r² threshold (higher = more SNPs retained)

# Directories
TEMP_DIR="${DATA_DIR}/temp"
REF_DIR="${TEMP_DIR}/1KGPHGDP"
SHARED_AOU_DIR="${SCRIPT_DIR}/../shared/data/AllofUs_V8"
SHARED_META_DIR="${SCRIPT_DIR}/../shared/data/Metadata"

echo "=========================================="
echo "  AoU-HGDP Data Preparation"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  CPU cores: ${CPU_CORES}"
echo "  Google project: ${GOOGLE_PROJECT:-'Not set'}"
echo "  CDR version: ${CDR_VERSION:-'Not set'}"
echo ""
echo "Filtering parameters:"
echo "  MAF threshold: ${MAF_THRESHOLD}"
echo "  Geno threshold: ${GENO_THRESHOLD}"
echo "  LD pruning: ${LD_WINDOW}kb window, step ${LD_STEP}, r²=${LD_R2}"
echo ""

# Create directories
mkdir -p "${DATA_DIR}" "${TEMP_DIR}" "${REF_DIR}"

# =============================================================================
# Find plink and plink2 binaries (needed throughout the script)
# =============================================================================
echo "=========================================="
echo "Checking for required tools"
echo "=========================================="
echo ""

# Find plink2
echo "Looking for plink2..."
PLINK2=""
if [[ -f "${PROJECT_ROOT}/bin/plink2" ]]; then
    PLINK2="${PROJECT_ROOT}/bin/plink2"
    echo "  ✓ Using plink2 from bin/plink2"
elif command -v plink2 &> /dev/null; then
    PLINK2="plink2"
    echo "  ✓ Using plink2 from PATH"
else
    echo "  ✗ plink2 not found!"
    echo ""
    echo "Please ensure plink2 is available via one of:"
    echo "  1. Run setup.sh to download to bin/plink2 (recommended)"
    echo "  2. Add plink2 to your PATH"
    echo ""
    exit 1
fi

if ! ${PLINK2} --version &> /dev/null; then
    echo "  ✗ plink2 found but not executable!"
    exit 1
fi

# Find plink (v1.9)
echo "Looking for plink (v1.9)..."
PLINK=""
if [[ -f "${PROJECT_ROOT}/bin/plink" ]]; then
    PLINK="${PROJECT_ROOT}/bin/plink"
    echo "  ✓ Using plink from bin/plink"
elif command -v plink &> /dev/null; then
    PLINK="plink"
    echo "  ✓ Using plink from PATH"
else
    echo "  ✗ plink (v1.9) not found!"
    echo ""
    echo "Please ensure plink is available via one of:"
    echo "  1. Run setup.sh to download to bin/plink (recommended)"
    echo "  2. Add plink to your PATH"
    echo ""
    exit 1
fi

if ! ${PLINK} --version &> /dev/null; then
    echo "  ✗ plink found but not executable!"
    exit 1
fi

echo ""

# =============================================================================
# STEP 1-5: Reference Data Processing (HGDP+1KGP)
#   1) Download
#   2) Extract
#   3) Fix BIM chromosome prefixes
#   4) Split by chromosome
# =============================================================================
echo "=========================================="
echo "Step 1-4: HGDP+1KGP Reference Data"
echo "=========================================="

TAR_PATH="${TEMP_DIR}/1KGPHGDP.tar.gz"

# 1. Download
if [[ ! -f "${TAR_PATH}" ]]; then
    echo "Downloading HGDP+1KGP reference data..."
    echo "  Source: ${REF_DATA_URL}"
    gsutil cp "${REF_DATA_URL}" "${TEMP_DIR}/"
else
    echo "  Reference data archive already exists"
fi

# 2. Extract
if [[ ! -f "${REF_DIR}/extractedChrAllUnpruned.bed" ]]; then
    echo "Extracting reference data..."
    tar -xf "${TAR_PATH}" --directory "${TEMP_DIR}/"
else
    echo "  Reference data already extracted"
fi

# 3. Fix BIM file (add 'chr' prefix)
BIM_PATH="${REF_DIR}/extractedChrAllUnpruned.bim"
ORIG_BIM_PATH="${REF_DIR}/extractedChrAllUnpruned.Original.bim"

if [[ ! -f "${ORIG_BIM_PATH}" ]]; then
    echo "Fixing BIM file chromosome prefixes..."
    mv "${BIM_PATH}" "${ORIG_BIM_PATH}"
    awk '{print "chr"$1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6}' "${ORIG_BIM_PATH}" > "${BIM_PATH}"
else
    echo "  BIM file already fixed"
fi

# 4. Split by chromosome (1-22)
echo "Splitting reference data by chromosome (using ${CPU_CORES} cores)..."
if [[ ! -f "${REF_DIR}/extractedChr22.bed" ]]; then
    # Export PLINK path for xargs subshell
    export PLINK
    seq 1 22 | xargs -I {} -P "${CPU_CORES}" sh -c \
        "${PLINK} --bfile ${REF_DIR}/extractedChrAllUnpruned \
               --keep-allele-order --allow-no-sex --chr {} \
               --make-bed --out ${REF_DIR}/extractedChr{}"
else
    echo "  Reference data already split by chromosome"
fi

echo "  ✓ HGDP+1KGP reference data ready"
echo ""

# =============================================================================
# STEP 5: All of Us Data Download (shared)
# =============================================================================
echo "=========================================="
echo "Step 5: All of Us Data Download"
echo "=========================================="
echo ""

# Check if shared AoU data is complete
AOU_DATA_COMPLETE=true

# Check for PLINK files
for ext in bed bim fam; do
    if [[ ! -f "${SHARED_AOU_DIR}/extractedChrAll.${ext}" ]]; then
        AOU_DATA_COMPLETE=false
        break
    fi
done

# Check for metadata files
if [[ ! -f "${SHARED_META_DIR}/DemographicData.tsv" ]]; then
    AOU_DATA_COMPLETE=false
fi

if [[ "$AOU_DATA_COMPLETE" == "true" ]]; then
    echo "✓ Shared AoU data already downloaded at: ${SHARED_AOU_DIR}"
    echo "  Skipping download (shared across all AoU experiments)"
else
    echo "Downloading AoU data to shared location..."
    echo "  This will be cached for all AoU experiments"
    bash "${SCRIPT_DIR}/../shared/download_aou_data.sh"
fi
echo ""

# =============================================================================
# STEP 6: Check overlap before filtering
# =============================================================================
echo "=========================================="
echo "Step 6: Check Overlap (before filtering)"
echo "=========================================="

AOU_UNFILTERED="${SHARED_AOU_DIR}/extractedChrAll"
awk '{print $1":"$4}' "${AOU_UNFILTERED}.bim" | sort > "${TEMP_DIR}/aou_unfiltered_pos.txt"
awk '{print $1":"$4}' "${REF_DIR}/extractedChrAllUnpruned.bim" | sort > "${TEMP_DIR}/hgdp_pos.txt"
OVERLAP_BEFORE=$(comm -12 "${TEMP_DIR}/aou_unfiltered_pos.txt" "${TEMP_DIR}/hgdp_pos.txt" | wc -l)
echo "  Overlap before filtering: ${OVERLAP_BEFORE}"
echo ""

# =============================================================================
# STEP 7: Filter AoU Data (indel removal + QC)
# =============================================================================
echo "=========================================="
echo "Step 7: Filter AoU Data"
echo "=========================================="

AOU_FILTERED="${TEMP_DIR}/aou_filtered"
if [[ ! -f "${AOU_FILTERED}.bed" ]]; then
    echo "  Filtering AoU (MAF > ${MAF_THRESHOLD}, missing < ${GENO_THRESHOLD}, no indels)..."

    # Get initial SNP count
    INITIAL_AOU_SNPS=$(wc -l < "${AOU_UNFILTERED}.bim")
    echo "  Initial AoU SNPs: $INITIAL_AOU_SNPS"

    # Step 1: Remove indels (keep only single-nucleotide biallelic SNPs)
    echo "  Removing indels (keeping only A/T/G/C SNPs)..."
    awk 'length($5) == 1 && length($6) == 1 && $5 ~ /^[ATGC]$/ && $6 ~ /^[ATGC]$/ {print $2}' "${AOU_UNFILTERED}.bim" > "${TEMP_DIR}/aou_snps_no_indels.txt"
    NO_INDEL_SNPS=$(wc -l < "${TEMP_DIR}/aou_snps_no_indels.txt")
    echo "    SNPs after indel removal: $NO_INDEL_SNPS"

    # Step 2: Apply quality filters with plink2
    ${PLINK2} --bfile ${AOU_UNFILTERED} \
        --extract ${TEMP_DIR}/aou_snps_no_indels.txt \
        --geno ${GENO_THRESHOLD} \
        --maf ${MAF_THRESHOLD} \
        --output-chr chrM \
        --memory 100000 \
        --make-bed \
        --out ${AOU_FILTERED}

    AOU_FILTERED_SNPS=$(wc -l < "${AOU_FILTERED}.bim")
    echo "  AoU SNPs after filtering: ${AOU_FILTERED_SNPS}"
else
    echo "  Using existing filtered AoU data"
    AOU_FILTERED_SNPS=$(wc -l < "${AOU_FILTERED}.bim")
    echo "  Filtered AoU SNPs: ${AOU_FILTERED_SNPS}"
fi
echo ""

# =============================================================================
# STEP 8: Filter HGDP Data (comprehensive filtering)
# =============================================================================
echo "=========================================="
echo "Step 8: Filter HGDP Data"
echo "=========================================="
echo ""
echo "This step applies comprehensive filtering to HGDP+1KGP:"
echo "  8a. Remove indels (keep only biallelic SNPs)"
echo "  8b. Remove duplicate/multi-allelic positions"
echo "  8c. Exclude GIAB difficult regions"
echo "  8d. Exclude HLA/MHC region"
echo "  8e. Apply MAF filter (>= ${MAF_THRESHOLD})"
echo "  8f. Apply missingness filter (< ${GENO_THRESHOLD})"
echo ""

HGDP_INPUT="${REF_DIR}/extractedChrAllUnpruned"
HGDP_FILTERED="${TEMP_DIR}/hgdp_filtered"
SHARED_DIR="${SCRIPT_DIR}/../_shared"

if [[ ! -f "${HGDP_FILTERED}.bed" ]]; then
    INITIAL_HGDP_SNPS=$(wc -l < "${HGDP_INPUT}.bim")
    echo "  Initial HGDP SNPs: $INITIAL_HGDP_SNPS"

    # -------------------------------------------------------------------------
    # 8a. Remove indels (keep only single-nucleotide biallelic SNPs)
    # -------------------------------------------------------------------------
    echo ""
    echo "  [8a] Removing indels (keeping only A/T/G/C SNPs)..."
    awk 'length($5) == 1 && length($6) == 1 && $5 ~ /^[ATGC]$/ && $6 ~ /^[ATGC]$/ {print $2}' \
        "${HGDP_INPUT}.bim" > "${TEMP_DIR}/hgdp_snps_no_indels.txt"
    NO_INDEL_COUNT=$(wc -l < "${TEMP_DIR}/hgdp_snps_no_indels.txt")
    echo "    SNPs after indel removal: $NO_INDEL_COUNT"

    # Create intermediate file without indels
    HGDP_NO_INDELS="${TEMP_DIR}/hgdp_no_indels"
    ${PLINK} --bfile ${HGDP_INPUT} \
        --extract ${TEMP_DIR}/hgdp_snps_no_indels.txt \
        --keep-allele-order \
        --make-bed \
        --out ${HGDP_NO_INDELS}

    # -------------------------------------------------------------------------
    # 8b. Remove duplicate/multi-allelic positions (keep lowest missingness, highest MAF)
    # -------------------------------------------------------------------------
    echo ""
    echo "  [8b] Removing duplicate/multi-allelic positions..."

    # Calculate missingness
    echo "    Calculating per-SNP missingness..."
    ${PLINK} --bfile ${HGDP_NO_INDELS} \
        --missing \
        --out ${TEMP_DIR}/hgdp_missing

    # Calculate allele frequencies
    echo "    Calculating allele frequencies..."
    ${PLINK} --bfile ${HGDP_NO_INDELS} \
        --freq \
        --out ${TEMP_DIR}/hgdp_freq

    # Run Python script to identify SNPs to keep
    echo "    Identifying best SNP per position..."
    python3 ${SHARED_DIR}/filter_duplicates.py \
        --bim ${HGDP_NO_INDELS}.bim \
        --lmiss ${TEMP_DIR}/hgdp_missing.lmiss \
        --frq ${TEMP_DIR}/hgdp_freq.frq \
        --out ${TEMP_DIR}/hgdp_dedup

    DEDUP_COUNT=$(wc -l < "${TEMP_DIR}/hgdp_dedup.keep_snps.txt")
    echo "    SNPs after deduplication: $DEDUP_COUNT"

    # Create intermediate file without duplicates
    HGDP_DEDUP="${TEMP_DIR}/hgdp_deduped"
    ${PLINK} --bfile ${HGDP_NO_INDELS} \
        --extract ${TEMP_DIR}/hgdp_dedup.keep_snps.txt \
        --keep-allele-order \
        --make-bed \
        --out ${HGDP_DEDUP}

    # -------------------------------------------------------------------------
    # 8c & 8d. Exclude GIAB difficult regions and HLA region
    # -------------------------------------------------------------------------
    echo ""
    echo "  [8c] Downloading GIAB difficult regions blacklist..."

    GIAB_URL="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/genome-stratifications/v3.6/GRCh38@all/Union/GRCh38_alldifficultregions.bed.gz"
    GIAB_BED="${TEMP_DIR}/GRCh38_alldifficultregions.bed"
    EXCLUDE_BED="${TEMP_DIR}/exclude_regions.bed"

    if [[ ! -f "${GIAB_BED}" ]]; then
        echo "    Downloading from NCBI..."
        curl -sL "${GIAB_URL}" | gunzip > "${GIAB_BED}"
        echo "    Downloaded $(wc -l < "${GIAB_BED}") regions"
    else
        echo "    GIAB blacklist already downloaded"
    fi

    echo ""
    echo "  [8d] Creating HLA/MHC exclusion region..."

    # HLA region (chr6:28.5-33.5Mb) - uses 'chr' prefix to match BIM
    HLA_BED="${TEMP_DIR}/HLA_region.bed"
    echo -e "chr6\t28510120\t33480577" > "${HLA_BED}"

    # Combine GIAB and HLA regions
    # Add 'chr' prefix to GIAB regions if needed (GIAB uses 'chr' prefix already for GRCh38)
    cat "${GIAB_BED}" "${HLA_BED}" > "${EXCLUDE_BED}"
    EXCLUDE_REGION_COUNT=$(wc -l < "${EXCLUDE_BED}")
    echo "    Total exclusion regions: $EXCLUDE_REGION_COUNT"

    # Exclude difficult regions
    echo "    Excluding difficult regions and HLA..."
    HGDP_NO_DIFFICULT="${TEMP_DIR}/hgdp_no_difficult"
    ${PLINK2} --bfile ${HGDP_DEDUP} \
        --exclude range ${EXCLUDE_BED} \
        --make-bed \
        --out ${HGDP_NO_DIFFICULT}

    NO_DIFFICULT_COUNT=$(wc -l < "${HGDP_NO_DIFFICULT}.bim")
    echo "    SNPs after excluding difficult regions: $NO_DIFFICULT_COUNT"

    # -------------------------------------------------------------------------
    # 8e & 8f. Apply MAF and missingness filters
    # -------------------------------------------------------------------------
    echo ""
    echo "  [8e/8f] Applying MAF (>= ${MAF_THRESHOLD}) and geno (< ${GENO_THRESHOLD}) filters..."

    ${PLINK2} --bfile ${HGDP_NO_DIFFICULT} \
        --maf ${MAF_THRESHOLD} \
        --geno ${GENO_THRESHOLD} \
        --output-chr chrM \
        --make-bed \
        --out ${HGDP_FILTERED}

    HGDP_FILTERED_SNPS=$(wc -l < "${HGDP_FILTERED}.bim")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    echo ""
    echo "  HGDP filtering summary:"
    echo "    Initial SNPs:                    $INITIAL_HGDP_SNPS"
    echo "    After indel removal:             $NO_INDEL_COUNT"
    echo "    After deduplication:             $DEDUP_COUNT"
    echo "    After excluding difficult/HLA:   $NO_DIFFICULT_COUNT"
    echo "    After MAF/geno filters:          $HGDP_FILTERED_SNPS"
else
    echo "  Using existing filtered HGDP data"
    HGDP_FILTERED_SNPS=$(wc -l < "${HGDP_FILTERED}.bim")
    echo "  Filtered HGDP SNPs: ${HGDP_FILTERED_SNPS}"
fi
echo ""

# =============================================================================
# STEP 9: Check overlap after filtering both datasets
# =============================================================================
echo "=========================================="
echo "Step 9: Check Overlap (after filtering)"
echo "=========================================="

awk '{print $1":"$4}' "${AOU_FILTERED}.bim" | sort > "${TEMP_DIR}/aou_filtered_pos.txt"
awk '{print $1":"$4}' "${HGDP_FILTERED}.bim" | sort > "${TEMP_DIR}/hgdp_filtered_pos.txt"
OVERLAP_AFTER=$(comm -12 "${TEMP_DIR}/aou_filtered_pos.txt" "${TEMP_DIR}/hgdp_filtered_pos.txt" | wc -l)
echo "  Overlap after filtering both: ${OVERLAP_AFTER}"
echo ""

# =============================================================================
# STEP 10: Standardize SNP IDs (pos:ref:alt)
# =============================================================================
echo "=========================================="
echo "Step 10: Standardize SNP IDs (pos:ref:alt)"
echo "=========================================="

# Standardize SNP IDs to pos:ref:alt format (handles chr naming differences)
echo "Standardizing SNP IDs to pos:ref:alt format..."

# Standardize HGDP (chr1:... → pos:ref:alt)
if [[ ! -f "${TEMP_DIR}/hgdp_standardized.bed" ]]; then
    echo "  Standardizing HGDP dataset (using filtered data)..."
    ${PLINK2} --bfile ${HGDP_FILTERED} \
        --set-all-var-ids '@:#:$r:$a' \
        --new-id-max-allele-len 100 missing \
        --make-bed \
        --out ${TEMP_DIR}/hgdp_standardized
else
    echo "  HGDP dataset already standardized"
fi

# Standardize AoU (chr1:... → pos:ref:alt)
if [[ ! -f "${TEMP_DIR}/aou_standardized.bed" ]]; then
    echo "  Standardizing AoU dataset (using filtered data)..."
    ${PLINK2} --bfile ${AOU_FILTERED} \
        --set-all-var-ids '@:#:$r:$a' \
        --new-id-max-allele-len 100 missing \
        --memory 100000 \
        --make-bed \
        --out ${TEMP_DIR}/aou_standardized
else
    echo "  AoU dataset already standardized"
fi

echo "  ✓ SNP IDs standardized (originals untouched)"

# =============================================================================
# STEP 11: LD prune reference and apply to AoU
# =============================================================================
echo ""
echo "=========================================="
echo "Step 11: LD prune reference and apply to AoU"
echo "=========================================="

# LD prune reference set and apply to AoU (keeps both datasets aligned and smaller)
echo "LD pruning reference SNPs and applying to AoU..."
echo "  Parameters: ${LD_WINDOW}kb window, step ${LD_STEP}, r²=${LD_R2}"
HGDP_PRUNE_PREFIX="${TEMP_DIR}/hgdp_prune"
HGDP_PRUNED_PREFIX="${TEMP_DIR}/hgdp_pruned"
AOU_PRUNED_PREFIX="${TEMP_DIR}/aou_pruned"

if [[ ! -f "${HGDP_PRUNE_PREFIX}.prune.in" ]]; then
    echo "  Computing prune list on HGDP reference..."
    ${PLINK2} --bfile ${TEMP_DIR}/hgdp_standardized \
        --indep-pairwise ${LD_WINDOW} ${LD_STEP} ${LD_R2} \
        --memory 100000 \
        --out ${HGDP_PRUNE_PREFIX}
else
    echo "  Prune list already exists"
fi

PRUNE_IN_COUNT=$(wc -l < "${HGDP_PRUNE_PREFIX}.prune.in")
echo "  SNPs retained after LD pruning: ${PRUNE_IN_COUNT}"

if [[ ! -f "${HGDP_PRUNED_PREFIX}.bed" ]]; then
    echo "  Applying prune list to HGDP reference..."
    ${PLINK2} --bfile ${TEMP_DIR}/hgdp_standardized \
        --extract ${HGDP_PRUNE_PREFIX}.prune.in \
        --make-bed \
        --out ${HGDP_PRUNED_PREFIX}
else
    echo "  HGDP pruned dataset already exists"
fi

if [[ ! -f "${AOU_PRUNED_PREFIX}.bed" ]]; then
    echo "  Applying prune list to AoU dataset..."
    ${PLINK2} --bfile ${TEMP_DIR}/aou_standardized \
        --extract ${HGDP_PRUNE_PREFIX}.prune.in \
        --memory 100000 \
        --make-bed \
        --out ${AOU_PRUNED_PREFIX}
else
    echo "  AoU pruned dataset already exists"
fi

# Update downstream inputs to use pruned datasets
HGDP_PLINK_PREFIX="${HGDP_PRUNED_PREFIX}"
AOU_PLINK_PREFIX="${AOU_PRUNED_PREFIX}"

# =============================================================================
# STEP 12: Find SNP intersection and check allele compatibility
# =============================================================================
echo ""
echo "=========================================="
echo "Step 12: Find SNP intersection"
echo "=========================================="

# Check if final common SNPs already computed
if [[ -f "${TEMP_DIR}/final_common_snps.txt" ]] && [[ -f "${TEMP_DIR}/flip_list.txt" ]] && [[ -f "${TEMP_DIR}/exclude_list.txt" ]]; then
    echo "  SNP intersection already computed, skipping..."
    FINAL_SNP_COUNT=$(wc -l < "${TEMP_DIR}/final_common_snps.txt")
    FLIP_COUNT=$(wc -l < "${TEMP_DIR}/flip_list.txt")
    EXCLUDE_COUNT=$(wc -l < "${TEMP_DIR}/exclude_list.txt")
    echo "  Final SNPs: $FINAL_SNP_COUNT, Flip: $FLIP_COUNT, Exclude: $EXCLUDE_COUNT"
else
    # Find SNP intersection
    echo "Finding SNP intersection..."

    awk '{print $2}' "${HGDP_PLINK_PREFIX}.bim" | sort > "${TEMP_DIR}/hgdp_snps.txt"
    awk '{print $2}' "${AOU_PLINK_PREFIX}.bim" | sort > "${TEMP_DIR}/aou_snps.txt"

    HGDP_SNP_COUNT=$(wc -l < "${TEMP_DIR}/hgdp_snps.txt")
    AOU_SNP_COUNT=$(wc -l < "${TEMP_DIR}/aou_snps.txt")

    echo "  Extracted SNP lists:"
    echo "    HGDP: $HGDP_SNP_COUNT SNPs"
    echo "    AoU: $AOU_SNP_COUNT SNPs"

    comm -12 "${TEMP_DIR}/hgdp_snps.txt" "${TEMP_DIR}/aou_snps.txt" > "${TEMP_DIR}/common_snps.txt"

    COMMON_SNP_COUNT=$(wc -l < "${TEMP_DIR}/common_snps.txt")
    echo "  ✓ Found $COMMON_SNP_COUNT common SNPs"

    if [[ $COMMON_SNP_COUNT -lt 50000 ]]; then
        echo "  ✗ Less than 50K common SNPs! Data may be incompatible."
        exit 1
    elif [[ $COMMON_SNP_COUNT -lt 100000 ]]; then
        echo "  ⚠ Less than 100K common SNPs! Check data compatibility."
    fi

    # Check allele consistency
    echo "Checking allele consistency..."

    python3 << EOF
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path("${PROJECT_ROOT}/src")))

from manifold_genetics.utils.io import check_allele_compatibility

# Read inputs (pruned BIM files)
hgdp_bim = Path("${HGDP_PLINK_PREFIX}.bim")
aou_bim = Path("${AOU_PLINK_PREFIX}.bim")
common_snps_file = Path("${TEMP_DIR}/common_snps.txt")
output_dir = Path("${TEMP_DIR}")

# Read common SNPs
with open(common_snps_file, 'r') as f:
    common_snps = {line.strip() for line in f if line.strip()}

# Check allele compatibility (IDs already standardized!)
exact_matches, need_flip, incompatible = check_allele_compatibility(
    hgdp_bim,
    aou_bim,
    common_snps
)

# Write flip/exclude lists
with open(output_dir / "flip_list.txt", 'w') as f:
    for snp in sorted(need_flip):
        f.write(f"{snp}\n")

with open(output_dir / "exclude_list.txt", 'w') as f:
    for snp in sorted(incompatible):
        f.write(f"{snp}\n")
EOF

    # Count results
    FLIP_COUNT=$(wc -l < "${TEMP_DIR}/flip_list.txt" 2>/dev/null || echo "0")
    EXCLUDE_COUNT=$(wc -l < "${TEMP_DIR}/exclude_list.txt" 2>/dev/null || echo "0")
    USABLE_SNPS=$((COMMON_SNP_COUNT - EXCLUDE_COUNT))

    echo "  ✓ Allele check complete:"
    echo "    SNPs to flip: $FLIP_COUNT"
    echo "    SNPs to exclude: $EXCLUDE_COUNT"
    echo "    Usable SNPs: $USABLE_SNPS"

    # Create final common SNPs list (excluding incompatible)
    if [[ $EXCLUDE_COUNT -gt 0 ]]; then
        echo "Removing excluded SNPs from common list..."
        comm -23 <(sort "${TEMP_DIR}/common_snps.txt") <(sort "${TEMP_DIR}/exclude_list.txt") > "${TEMP_DIR}/final_common_snps.txt"
    else
        cp "${TEMP_DIR}/common_snps.txt" "${TEMP_DIR}/final_common_snps.txt"
    fi

    FINAL_SNP_COUNT=$(wc -l < "${TEMP_DIR}/final_common_snps.txt")
    echo "  ✓ Final SNP count: $FINAL_SNP_COUNT"
fi

# =============================================================================
# STEP 13: Create intersected PLINK files
# =============================================================================
echo ""
echo "=========================================="
echo "Step 13: Create final datasets"
echo "=========================================="

# Create intersected PLINK files
echo "Creating intersected PLINK files..."

# Create HGDP intersected dataset (from pruned files)
if [[ ! -f "${TEMP_DIR}/hgdp_intersected.bed" ]]; then
    echo "  Creating HGDP intersected dataset..."
    HGDP_CMD="${PLINK2} --bfile ${HGDP_PLINK_PREFIX} --extract ${TEMP_DIR}/final_common_snps.txt"
    if [[ $FLIP_COUNT -gt 0 ]]; then
        HGDP_CMD="$HGDP_CMD --flip ${TEMP_DIR}/flip_list.txt"
    fi
    HGDP_CMD="$HGDP_CMD --make-bed --out ${TEMP_DIR}/hgdp_intersected"
    eval $HGDP_CMD
else
    echo "  HGDP intersected dataset already exists"
fi

# Create AoU intersected dataset (from pruned files)
if [[ ! -f "${TEMP_DIR}/aou_intersected.bed" ]]; then
    echo "  Creating AoU intersected dataset..."
    ${PLINK2} --bfile ${AOU_PLINK_PREFIX} \
        --extract ${TEMP_DIR}/final_common_snps.txt \
        --memory 100000 \
        --make-bed \
        --out ${TEMP_DIR}/aou_intersected
else
    echo "  AoU intersected dataset already exists"
fi

echo "  ✓ Intersected datasets created"

# Create final processed subsets
echo "Creating final processed subsets..."

# Create HGDP fit subset (using all intersected HGDP samples)
if [[ ! -f "${DATA_DIR}/fit_subset.bed" ]]; then
    cp "${TEMP_DIR}/hgdp_intersected.bed" "${DATA_DIR}/fit_subset.bed"
    cp "${TEMP_DIR}/hgdp_intersected.bim" "${DATA_DIR}/fit_subset.bim"
    cp "${TEMP_DIR}/hgdp_intersected.fam" "${DATA_DIR}/fit_subset.fam"
else
    echo "  fit_subset already exists, skipping"
fi

# Create AoU project subset (using all intersected AoU samples)
if [[ ! -f "${DATA_DIR}/project_subset.bed" ]]; then
    cp "${TEMP_DIR}/aou_intersected.bed" "${DATA_DIR}/project_subset.bed"
    cp "${TEMP_DIR}/aou_intersected.bim" "${DATA_DIR}/project_subset.bim"
    cp "${TEMP_DIR}/aou_intersected.fam" "${DATA_DIR}/project_subset.fam"
else
    echo "  project_subset already exists, skipping"
fi

echo "  ✓ Final subsets created:"

# Get final sample counts
FIT_SAMPLES=$(wc -l < "${DATA_DIR}/fit_subset.fam")
PROJECT_SAMPLES=$(wc -l < "${DATA_DIR}/project_subset.fam")

echo "    Fit subset (HGDP): $FIT_SAMPLES samples"
echo "    Project subset (AoU): $PROJECT_SAMPLES samples"
echo "    Common SNPs: $FINAL_SNP_COUNT SNPs"
echo ""

# =============================================================================
# STEP 14: Create Labels and Colormaps
# =============================================================================
echo "=========================================="
echo "Step 14: Labels and Colormaps"
echo "=========================================="

# Create HGDP labels
echo "Creating HGDP labels..."

# Check if HGDP labels already exist
if [[ -f "${DATA_DIR}/hgdp_labels.csv" ]]; then
    echo "  HGDP labels already exist, skipping..."
    HGDP_LABEL_COUNT=$(wc -l < "${DATA_DIR}/hgdp_labels.csv")
    echo "  Existing labels: $((HGDP_LABEL_COUNT - 1)) samples"
else
    # Create HGDP labels directly from .fam file
    # The FID (column 1) contains the Population label (e.g., ACB, GBR, etc.)
    # The IID (column 2) contains the sample_id
    python3 << EOF
import pandas as pd

print("Creating HGDP labels from .fam file...")
fam = pd.read_csv("${DATA_DIR}/fit_subset.fam", sep=r'\s+', header=None,
                  names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Phenotype'])

# Create labels DataFrame with sample_id and Population
hgdp_labels = pd.DataFrame({
    'sample_id': fam['IID'].astype(str),
    'Population': fam['FID'].astype(str)
})

# Clean up population names: strip "forReference" prefix
hgdp_labels['Population'] = hgdp_labels['Population'].str.replace('^forReference', '', regex=True)

# Save labels
hgdp_labels.to_csv("${DATA_DIR}/hgdp_labels.csv", index=False)
print(f"  ✓ Saved HGDP labels: {len(hgdp_labels)} samples")
print(f"  Unique populations: {hgdp_labels['Population'].nunique()}")
print(f"  Populations: {sorted(hgdp_labels['Population'].unique())}")
EOF
fi

# Create AoU labels
echo "Creating AoU labels..."

# Check if AoU labels already exist
if [[ -f "${DATA_DIR}/aou_labels.csv" ]]; then
    echo "  AoU labels already exist, skipping..."
    AOU_LABEL_COUNT=$(wc -l < "${DATA_DIR}/aou_labels.csv")
    echo "  Existing labels: $((AOU_LABEL_COUNT - 1)) samples"
else
    # Check if AoU metadata exists
    AOU_METADATA="${SHARED_META_DIR}/DemographicData.tsv"

    if [[ -f "$AOU_METADATA" ]]; then
        python3 << EOF
import pandas as pd

# Read AoU metadata
print("Reading AoU demographic data...")
aou_metadata = pd.read_csv("${AOU_METADATA}", sep='\t', low_memory=False)
print(f"  Total metadata records: {len(aou_metadata)}")

# Read AoU .fam file to get available samples
fam = pd.read_csv("${DATA_DIR}/project_subset.fam", sep=r'\s+', header=None,
                  names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Phenotype'])
available_samples = set(fam['IID'].astype(str))
print(f"  Available in intersected data: {len(available_samples)}")

# Filter metadata to only include samples we have
aou_metadata['sample_id'] = aou_metadata['person_id'].astype(str)
aou_labels = aou_metadata[aou_metadata['sample_id'].isin(available_samples)].copy()

# Select relevant columns (adjust based on what's in your metadata)
columns_to_keep = ['sample_id']
if 'race' in aou_labels.columns:
    columns_to_keep.append('race')
if 'ethnicity' in aou_labels.columns:
    columns_to_keep.append('ethnicity')
if 'race_ethnicity' in aou_labels.columns:
    columns_to_keep.append('race_ethnicity')

aou_labels = aou_labels[columns_to_keep]

# Save labels
aou_labels.to_csv("${DATA_DIR}/aou_labels.csv", index=False)
print(f"  ✓ Saved AoU labels: {len(aou_labels)} samples")
print(f"  Columns: {', '.join(columns_to_keep)}")

if len(aou_labels) < len(available_samples):
    missing = len(available_samples) - len(aou_labels)
    print(f"  Warning: {missing} samples in intersected data have no metadata")
EOF
else
    echo ""
    echo "  ✗ ERROR: AoU metadata not found at: $AOU_METADATA"
    echo ""
    echo "  This file is created by the AoU data download script."
    echo "  Ensure WORKSPACE_CDR is set and re-run:"
    echo ""
    echo "    bash ${PROJECT_ROOT}/examples/aou/shared/download_aou_data.sh"
    echo ""
    echo "  Then re-run this script."
    exit 1
    fi
fi

echo ""
echo "  ✓ Label files created"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=========================================="
echo "Data Preparation Complete!"
echo "=========================================="
echo ""
echo "✓ All steps completed successfully:"
echo "  ✓ Step 1-4: HGDP+1KGP reference data downloaded and split"
echo "  ✓ Step 5: AoU genotype data downloaded"
echo "  ✓ Step 6: Overlap before filtering (${OVERLAP_BEFORE} positions)"
echo "  ✓ Step 7: AoU data filtered (MAF>${MAF_THRESHOLD}, geno<${GENO_THRESHOLD}, no indels)"
echo "  ✓ Step 8: HGDP data filtered (comprehensive):"
echo "      - 8a: Indels removed"
echo "      - 8b: Duplicate/multi-allelic positions removed"
echo "      - 8c: GIAB difficult regions excluded"
echo "      - 8d: HLA/MHC region excluded"
echo "      - 8e/8f: MAF>${MAF_THRESHOLD}, geno<${GENO_THRESHOLD}"
echo "  ✓ Step 9: Overlap after filtering (${OVERLAP_AFTER} positions)"
echo "  ✓ Step 10: SNP IDs standardized"
echo "  ✓ Step 11: LD pruning (${LD_WINDOW}kb, r²=${LD_R2})"
echo "  ✓ Step 12: SNP intersection found"
echo "  ✓ Step 13: Final datasets created"
echo "  ✓ Step 14: Labels created"
echo ""
echo "Generated files in ${DATA_DIR}:"
echo "  📊 Processed PLINK data:"
echo "    - fit_subset.{bed,bim,fam}     (HGDP+1KGP, $FIT_SAMPLES samples, $FINAL_SNP_COUNT SNPs)"
echo "    - project_subset.{bed,bim,fam} (AoU, $PROJECT_SAMPLES samples, $FINAL_SNP_COUNT SNPs)"
echo ""
echo "  🏷️  Sample labels:"
echo "    - hgdp_labels.csv    (HGDP sample metadata)"
echo "    - aou_labels.csv     (AoU sample metadata)"
echo ""
echo "  📋 Intermediate files (in data/temp/):"
echo "    - 1KGPHGDP.tar.gz (downloaded reference archive)"
echo "    - 1KGPHGDP/ (extracted reference data)"
echo "    - aou_filtered.{bed,bim,fam} (filtered AoU data)"
echo "    - hgdp_no_indels.{bed,bim,fam} (HGDP without indels)"
echo "    - hgdp_deduped.{bed,bim,fam} (HGDP without duplicates)"
echo "    - hgdp_no_difficult.{bed,bim,fam} (HGDP without GIAB/HLA)"
echo "    - hgdp_filtered.{bed,bim,fam} (final filtered HGDP data)"
echo "    - GRCh38_alldifficultregions.bed (GIAB blacklist)"
echo "    - *_standardized.{bed,bim,fam} (SNP IDs standardized)"
echo "    - *_pruned.{bed,bim,fam} (LD-pruned data)"
echo "    - common_snps.txt, flip_list.txt, exclude_list.txt"
echo ""
echo "Filtering parameters used:"
echo "  • MAF threshold: ${MAF_THRESHOLD}"
echo "  • Geno threshold: ${GENO_THRESHOLD}"
echo "  • LD pruning: ${LD_WINDOW}kb window, step ${LD_STEP}, r²=${LD_R2}"
echo ""
echo "To adjust SNP count, modify these environment variables and re-run:"
echo "  MAF_THRESHOLD=0.01 GENO_THRESHOLD=0.01 LD_R2=0.2 bash prepare_data.sh"
echo ""
echo "Next step: Run the cross-projection pipeline"
echo "  bash run_pipeline.sh"
echo ""
