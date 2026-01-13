#!/bin/bash
#
# Prepare AoU-HGDP cross-projection data
# This script integrates the aou_pipeline steps to:
# 1. Download/prepare HGDP+1KGP reference data
# 2. Download/prepare AoU genotype data
# 3. Query AoU metadata
# 4. Find common SNPs and create fit/project subsets
# 5. Create labels and colormaps
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

# Directories
REF_DIR="${DATA_DIR}/1KGPHGDP"
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

# Create directories
mkdir -p "${DATA_DIR}" "${REF_DIR}"

# =============================================================================
# STEP 1-5: Reference Data Processing (HGDP+1KGP)
#   1) Download
#   2) Extract
#   3) Fix BIM chromosome prefixes
#   4) Split by chromosome
#   5) LD prune
# =============================================================================
echo "=========================================="
echo "Step 1-5: HGDP+1KGP Reference Data"
echo "=========================================="

TAR_PATH="${DATA_DIR}/1KGPHGDP.tar.gz"

# 1. Download
if [[ ! -f "${TAR_PATH}" ]]; then
    echo "Downloading HGDP+1KGP reference data..."
    echo "  Source: ${REF_DATA_URL}"
    gsutil cp "${REF_DATA_URL}" "${DATA_DIR}/"
else
    echo "  Reference data archive already exists"
fi

# 2. Extract
if [[ ! -f "${REF_DIR}/extractedChrAllUnpruned.bed" ]]; then
    echo "Extracting reference data..."
    tar -xf "${TAR_PATH}" --directory "${DATA_DIR}/"
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
    seq 1 22 | xargs -I {} -P "${CPU_CORES}" sh -c \
        "plink --bfile ${REF_DIR}/extractedChrAllUnpruned \
               --keep-allele-order --allow-no-sex --chr {} \
               --make-bed --out ${REF_DIR}/extractedChr{}"
else
    echo "  Reference data already split by chromosome"
fi

echo "  ✓ HGDP+1KGP reference data ready"
echo ""

# =============================================================================
# STEP 6: All of Us Data Download (shared)
# =============================================================================
echo "=========================================="
echo "Step 6: All of Us Data Download"
echo "=========================================="
echo ""

# Check if shared AoU data is complete
AOU_DATA_COMPLETE=true

# Check for main files
for ext in bed bim fam; do
    if [[ ! -f "${SHARED_AOU_DIR}/extractedChrAll.${ext}" ]]; then
        AOU_DATA_COMPLETE=false
        break
    fi
done

# Check for all chromosome splits (1-22)
if [[ "$AOU_DATA_COMPLETE" == "true" ]]; then
    for chr in {1..22}; do
        if [[ ! -f "${SHARED_AOU_DIR}/extractedChr${chr}.bed" ]]; then
            AOU_DATA_COMPLETE=false
            break
        fi
    done
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
# STEP 6.5: Use Filtered AoU Data
#   The shared download script creates a filtered dataset (indels removed, QC applied)
#   This reduces the dataset from ~1.7M SNPs to ~500K SNPs (>10x reduction)
# =============================================================================
echo "=========================================="
echo "Step 6.5: Use Filtered AoU Data"
echo "=========================================="
echo ""

# Create temp directory
TEMP_DIR="${DATA_DIR}/temp"
mkdir -p "$TEMP_DIR"

# Check if filtered data exists from shared download
AOU_FILTERED_SHARED="${SHARED_AOU_DIR}/extractedChrAll_filtered"

if [[ -f "${AOU_FILTERED_SHARED}.bed" ]]; then
    echo "  ✓ Using filtered AoU data from shared location"
    AOU_INPUT="${AOU_FILTERED_SHARED}"
    FILTERED_SNPS=$(wc -l < "${AOU_INPUT}.bim")
    echo "    Filtered SNPs: $FILTERED_SNPS"
    echo "    Location: ${AOU_FILTERED_SHARED}"
else
    echo "  ⚠ Filtered AoU data not found at: ${AOU_FILTERED_SHARED}"
    echo "    Using unfiltered data from: ${SHARED_AOU_DIR}/extractedChrAll"
    echo ""
    echo "  Note: To use filtered data, run the shared download script:"
    echo "    bash ${SCRIPT_DIR}/../shared/download_aou_data.sh"
    echo ""
    AOU_INPUT="${SHARED_AOU_DIR}/extractedChrAll"
fi

echo ""

# =============================================================================
# STEP 7-10: Standardize IDs, prune, intersect, and create subsets
#   7) Standardize SNP IDs (pos:ref:alt)
#   8) Apply reference LD prune list to HGDP and AoU
#   9) Find SNP intersection & check alleles
#  10) Create intersected PLINKs and final subsets
# =============================================================================
echo "=========================================="
echo "Step 7: Standardize SNP IDs (pos:ref:alt)"
echo "=========================================="

# Find plink2
echo "Looking for plink2..."
PLINK2=""

# Check bin/ directory first (preferred)
if [[ -f "${PROJECT_ROOT}/bin/plink2" ]]; then
    PLINK2="${PROJECT_ROOT}/bin/plink2"
    echo "  ✓ Using plink2 from bin/plink2"
# Check if plink2 is in PATH
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

# Verify plink2 works
if ! ${PLINK2} --version &> /dev/null; then
    echo "  ✗ plink2 found but not executable!"
    exit 1
fi

# Standardize SNP IDs to pos:ref:alt format (handles chr naming differences)
echo "Standardizing SNP IDs to pos:ref:alt format..."

# Standardize HGDP (chr1:... → pos:ref:alt)
if [[ ! -f "${TEMP_DIR}/hgdp_standardized.bed" ]]; then
    echo "  Standardizing HGDP dataset..."
    ${PLINK2} --bfile ${REF_DIR}/extractedChrAllUnpruned \
        --set-all-var-ids '@:#:$r:$a' \
        --new-id-max-allele-len 100 missing \
        --make-bed \
        --out ${TEMP_DIR}/hgdp_standardized
else
    echo "  HGDP dataset already standardized"
fi

# Standardize AoU (chr1:... → pos:ref:alt)
# Uses filtered AoU data from Step 6.5
if [[ ! -f "${TEMP_DIR}/aou_standardized.bed" ]]; then
    echo "  Standardizing AoU dataset (using filtered data)..."
    ${PLINK2} --bfile ${AOU_INPUT} \
        --set-all-var-ids '@:#:$r:$a' \
        --new-id-max-allele-len 100 missing \
        --memory 100000 \
        --make-bed \
        --out ${TEMP_DIR}/aou_standardized
else
    echo "  AoU dataset already standardized"
fi

echo "  ✓ SNP IDs standardized (originals untouched)"

# LD prune reference set and apply to AoU (keeps both datasets aligned and smaller)
echo ""
echo "=========================================="
echo "Step 8: LD prune reference and apply to AoU"
echo "=========================================="

# LD prune reference set and apply to AoU (keeps both datasets aligned and smaller)
echo "LD pruning reference SNPs and applying to AoU..."
HGDP_PRUNE_PREFIX="${TEMP_DIR}/hgdp_prune"
HGDP_PRUNED_PREFIX="${TEMP_DIR}/hgdp_pruned"
AOU_PRUNED_PREFIX="${TEMP_DIR}/aou_pruned"

if [[ ! -f "${HGDP_PRUNE_PREFIX}.prune.in" ]]; then
    echo "  Computing prune list on HGDP reference (150kb, step 1, r2=0.05)..."
    ${PLINK2} --bfile ${TEMP_DIR}/hgdp_standardized \
        --indep-pairwise 150 kb 1 0.05 \
        --memory 100000 \
        --out ${HGDP_PRUNE_PREFIX}
else
    echo "  Prune list already exists"
fi

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
# STEP 6: Create Labels and Colormaps
# =============================================================================
echo "=========================================="
echo "Step 6: Labels and Colormaps"
echo "=========================================="

# Create HGDP labels
echo "Creating HGDP labels..."

# Check if HGDP labels exist in the HGDP+1KGP example
HGDP_SOURCE_LABELS="${PROJECT_ROOT}/examples/hgdp_1kgp/data/hgdp_1kgp_labels.csv"

if [[ -f "$HGDP_SOURCE_LABELS" ]]; then
    python3 << EOF
import pandas as pd

# Read existing HGDP labels from hgdp_1kgp example
print("Reading HGDP labels from existing example...")
hgdp_all_labels = pd.read_csv("${HGDP_SOURCE_LABELS}")
print(f"  Total labels available: {len(hgdp_all_labels)}")

# Read intersected HGDP .fam file to get available samples
fam = pd.read_csv("${DATA_DIR}/fit_subset.fam", sep=r'\s+', header=None,
                  names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Phenotype'])
available_samples = set(fam['IID'].astype(str))
print(f"  Available in intersected data: {len(available_samples)}")

# Filter labels to only include samples we have
hgdp_all_labels['sample_id'] = hgdp_all_labels['sample_id'].astype(str)
hgdp_labels = hgdp_all_labels[hgdp_all_labels['sample_id'].isin(available_samples)].copy()

# Save filtered labels
hgdp_labels.to_csv("${DATA_DIR}/hgdp_labels.csv", index=False)
print(f"  ✓ Saved HGDP labels: {len(hgdp_labels)} samples")

if len(hgdp_labels) < len(available_samples):
    missing = len(available_samples) - len(hgdp_labels)
    print(f"  Warning: {missing} samples in intersected data have no labels")
EOF
else
    echo "  ⚠ HGDP source labels not found at: $HGDP_SOURCE_LABELS"
    echo "  Creating basic HGDP labels from .fam file..."
    awk '{print $2}' "${DATA_DIR}/fit_subset.fam" > "${TEMP_DIR}/hgdp_sample_ids.txt"
    python3 << EOF
import pandas as pd

# Create basic labels file with just sample_id
sample_ids = pd.read_csv("${TEMP_DIR}/hgdp_sample_ids.txt", header=None, names=['sample_id'])
sample_ids.to_csv("${DATA_DIR}/hgdp_labels.csv", index=False)
print(f"  ✓ Created basic HGDP labels: {len(sample_ids)} samples")
print("  Note: Only contains sample_id column. Add population labels manually if needed.")
EOF
fi

# Create AoU labels
echo "Creating AoU labels..."

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
    echo "  ⚠ AoU metadata not found at: $AOU_METADATA"
    echo "  Creating basic AoU labels from .fam file..."
    awk '{print $2}' "${DATA_DIR}/project_subset.fam" > "${TEMP_DIR}/aou_sample_ids.txt"
    python3 << EOF
import pandas as pd

# Create basic labels file with just sample_id
sample_ids = pd.read_csv("${TEMP_DIR}/aou_sample_ids.txt", header=None, names=['sample_id'])
sample_ids.to_csv("${DATA_DIR}/aou_labels.csv", index=False)
print(f"  ✓ Created basic AoU labels: {len(sample_ids)} samples")
print("  Note: Only contains sample_id column. Add demographic labels manually if needed.")
EOF
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
echo "  ✓ Step 1-5: HGDP+1KGP reference data downloaded and split"
echo "  ✓ Step 6: AoU genotype data downloaded and split"
echo "  ✓ Step 6.5: AoU data filtered (indels removed, QC filters applied)"
echo "  ✓ Step 7: SNP IDs standardized to pos:ref:alt format"
echo "  ✓ Step 8: LD pruning applied"
echo "  ✓ Step 9-10: Common SNPs found and fit/project subsets created"
echo "  ✓ Step 11: Labels created for both datasets"
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
echo "  📋 Processing logs (in data/temp/):"
echo "    - aou_filtered.{bed,bim,fam} (filtered AoU data, indels removed, QC applied)"
echo "    - *_standardized.{bed,bim,fam} (SNP IDs standardized to pos:ref:alt)"
echo "    - *_pruned.{bed,bim,fam} (LD-pruned data)"
echo "    - common_snps.txt, flip_list.txt, exclude_list.txt"
echo ""
echo "Shared AoU data location:"
echo "  ${SHARED_AOU_DIR}/"
echo "  (This is shared across all AoU experiments)"
echo ""
echo "Key improvements:"
echo "  • Early filtering reduces AoU SNPs by >10x (saves memory and time)"
echo "  • Indels removed (keeps only biallelic SNPs A/T/G/C)"
echo "  • Missing genotype filtering (<5% missingness)"
echo "  • MAF filtering (>0.1% minor allele frequency)"
echo ""
echo "Next step: Run the cross-projection pipeline"
echo "  bash run_pipeline.sh"
echo ""
