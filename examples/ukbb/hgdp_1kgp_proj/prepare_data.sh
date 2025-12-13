#!/bin/bash
#
# Data preparation for UKBB-HGDP cross-projection analysis
#
# This script:
# 1. Reads raw UKBB and HGDP PLINK files from mappings.json
# 2. Finds SNP intersection between datasets
# 3. Checks allele consistency and creates flip lists
# 4. Creates intersected PLINK subsets
# 5. Generates sample subsets and label files
#
# Usage:
#   cd /path/to/manifold_genetics/examples/ukbb/hgdp_1kgp_proj/
#   bash prepare_data.sh
#

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ANSI colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Print header
echo ""
echo "========================================="
echo "  UKBB-HGDP Cross-Projection Data Prep"
echo "========================================="
echo ""

# Paths
DATA_DIR="${SCRIPT_DIR}/data"
MAPPINGS_FILE="${DATA_DIR}/mappings.json"
TEMP_DIR="${DATA_DIR}/temp"

# Create temp directory
mkdir -p "$TEMP_DIR"

# Step 1: Read mappings file
print_status "Reading data paths from mappings.json..."

if [[ ! -f "$MAPPINGS_FILE" ]]; then
    print_error "Mappings file not found: $MAPPINGS_FILE"
    echo "Please create mappings.json with data paths"
    exit 1
fi

# Extract paths using Python (resolve relative paths from PROJECT_ROOT)
python3 << EOF > "${TEMP_DIR}/paths.sh"
import json
from pathlib import Path

with open('$MAPPINGS_FILE') as f:
    mappings = json.load(f)

project_root = Path('$PROJECT_ROOT')

# Resolve paths (relative ones get resolved from project root)
for key, value in mappings.items():
    if value.startswith('/'):
        # Absolute path
        print(f"{key.upper()}='{value}'")
    else:
        # Relative path - resolve from project root
        resolved = project_root / value
        print(f"{key.upper()}='{resolved}'")
EOF

source "${TEMP_DIR}/paths.sh"

print_success "Loaded paths:"
echo "  UKBB PLINK: $UKBB_PLINK"
echo "  HGDP PLINK: $HGDP_PLINK"
echo "  UKBB Metadata: $UKBB_METADATA"
echo "  HGDP Metadata: $HGDP_METADATA"

# Step 2: Verify input files exist
print_status "Verifying input files exist..."

REQUIRED_FILES=(
    "${UKBB_PLINK}.bed"
    "${UKBB_PLINK}.bim"
    "${UKBB_PLINK}.fam"
    "${HGDP_PLINK}.bed"
    "${HGDP_PLINK}.bim"
    "${HGDP_PLINK}.fam"
    "${HGDP_METADATA}"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        MISSING_FILES+=("$file")
    fi
done

if [[ ${#MISSING_FILES[@]} -gt 0 ]]; then
    print_error "Missing required files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

print_success "All required files found"

# Step 3: Find plink2 (need it early now)
print_status "Looking for plink2..."

PLINK2=""

# Check bin/ directory first (preferred)
if [[ -f "${PROJECT_ROOT}/bin/plink2" ]]; then
    PLINK2="${PROJECT_ROOT}/bin/plink2"
    print_success "Using plink2 from bin/plink2"
# Check if plink2 module is loaded
elif module list 2>&1 | grep -q plink2; then
    PLINK2="plink2"
    print_success "Using plink2 from loaded module"
# Check PATH
elif command -v plink2 &> /dev/null; then
    PLINK2="plink2"
    print_success "Using plink2 from PATH"
else
    print_error "plink2 not found!"
    echo ""
    echo "Please ensure plink2 is available via one of:"
    echo "  1. Run setup.sh to download to bin/plink2 (recommended)"
    echo "  2. Load plink2 module: module load plink2"
    echo "  3. Add plink2 to your PATH"
    echo ""
    exit 1
fi

# Verify plink2 works
if ! ${PLINK2} --version &> /dev/null; then
    print_error "plink2 found but not executable!"
    exit 1
fi

# Step 4: Standardize SNP IDs to pos:ref:alt format (handles chr naming differences)
print_status "Standardizing SNP IDs to pos:ref:alt format..."

# Standardize UKBB (chr1:... → pos:ref:alt)
print_status "  Standardizing UKBB dataset..."
${PLINK2} --bfile ${UKBB_PLINK} \
    --set-all-var-ids '@:#:$r:$a' \
    --new-id-max-allele-len 100 missing \
    --make-bed \
    --out ${TEMP_DIR}/ukbb_standardized

# Standardize HGDP (1:... → pos:ref:alt)
print_status "  Standardizing HGDP dataset..."
${PLINK2} --bfile ${HGDP_PLINK} \
    --set-all-var-ids '@:#:$r:$a' \
    --new-id-max-allele-len 100 missing \
    --make-bed \
    --out ${TEMP_DIR}/hgdp_standardized

print_success "SNP IDs standardized (originals untouched)"

# Step 5: Find SNP intersection (now both use same format!)
print_status "Finding SNP intersection..."

awk '{print $2}' "${TEMP_DIR}/ukbb_standardized.bim" | sort > "${TEMP_DIR}/ukbb_snps.txt"
awk '{print $2}' "${TEMP_DIR}/hgdp_standardized.bim" | sort > "${TEMP_DIR}/hgdp_snps.txt"

UKBB_SNP_COUNT=$(wc -l < "${TEMP_DIR}/ukbb_snps.txt")
HGDP_SNP_COUNT=$(wc -l < "${TEMP_DIR}/hgdp_snps.txt")

print_success "Extracted SNP lists:"
echo "  UKBB: $UKBB_SNP_COUNT SNPs"
echo "  HGDP: $HGDP_SNP_COUNT SNPs"

comm -12 "${TEMP_DIR}/ukbb_snps.txt" "${TEMP_DIR}/hgdp_snps.txt" > "${TEMP_DIR}/common_snps.txt"

COMMON_SNP_COUNT=$(wc -l < "${TEMP_DIR}/common_snps.txt")
print_success "Found $COMMON_SNP_COUNT common SNPs"

if [[ $COMMON_SNP_COUNT -lt 50000 ]]; then
    print_error "Less than 50K common SNPs! Data may be incompatible."
    exit 1
elif [[ $COMMON_SNP_COUNT -lt 100000 ]]; then
    print_warning "Less than 100K common SNPs! Check data compatibility."
fi

# Step 6: Check allele consistency
print_status "Checking allele consistency..."

python3 << EOF
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path("${PROJECT_ROOT}/src")))

from manifold_genetics.utils.io import check_allele_compatibility

# Read inputs (standardized BIM files)
ukbb_bim = Path("${TEMP_DIR}/ukbb_standardized.bim")
hgdp_bim = Path("${TEMP_DIR}/hgdp_standardized.bim")
common_snps_file = Path("${TEMP_DIR}/common_snps.txt")
output_dir = Path("${TEMP_DIR}")

# Read common SNPs
with open(common_snps_file, 'r') as f:
    common_snps = {line.strip() for line in f if line.strip()}

# Check allele compatibility (IDs already standardized!)
exact_matches, need_flip, incompatible = check_allele_compatibility(
    ukbb_bim,
    hgdp_bim,
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

print_success "Allele check complete:"
echo "  SNPs to flip: $FLIP_COUNT"
echo "  SNPs to exclude: $EXCLUDE_COUNT"
echo "  Usable SNPs: $USABLE_SNPS"

# Create final common SNPs list (excluding incompatible)
if [[ $EXCLUDE_COUNT -gt 0 ]]; then
    print_status "Removing excluded SNPs from common list..."
    comm -23 <(sort "${TEMP_DIR}/common_snps.txt") <(sort "${TEMP_DIR}/exclude_list.txt") > "${TEMP_DIR}/final_common_snps.txt"
else
    cp "${TEMP_DIR}/common_snps.txt" "${TEMP_DIR}/final_common_snps.txt"
fi

FINAL_SNP_COUNT=$(wc -l < "${TEMP_DIR}/final_common_snps.txt")
print_success "Final SNP count: $FINAL_SNP_COUNT"

# Step 8: Create intersected PLINK files
print_status "Creating intersected PLINK files..."

# Create UKBB intersected dataset (from standardized files)
print_status "Creating UKBB intersected dataset..."
UKBB_CMD="${PLINK2} --bfile ${TEMP_DIR}/ukbb_standardized --extract ${TEMP_DIR}/final_common_snps.txt"
if [[ $FLIP_COUNT -gt 0 ]]; then
    UKBB_CMD="$UKBB_CMD --flip ${TEMP_DIR}/flip_list.txt"
fi
UKBB_CMD="$UKBB_CMD --make-bed --out ${TEMP_DIR}/ukbb_intersected"

eval $UKBB_CMD

# Create HGDP intersected dataset (from standardized files)
print_status "Creating HGDP intersected dataset..."
${PLINK2} --bfile ${TEMP_DIR}/hgdp_standardized \
    --extract ${TEMP_DIR}/final_common_snps.txt \
    --make-bed \
    --out ${TEMP_DIR}/hgdp_intersected

print_success "Intersected datasets created"

# Step 9: Save UKBB sample list for user
print_status "Saving UKBB sample list..."

# Extract sample IDs from intersected UKBB data (FID IID format for plink2 --keep)
awk '{print $1, $2}' "${TEMP_DIR}/ukbb_intersected.fam" > "${DATA_DIR}/ukbb_samples.txt"
UKBB_SAMPLE_COUNT=$(wc -l < "${DATA_DIR}/ukbb_samples.txt")

print_success "Saved ${UKBB_SAMPLE_COUNT} UKBB sample IDs to ukbb_samples.txt"

# Create HGDP labels from existing labels file
print_status "Creating HGDP labels..."

python3 << EOF
import pandas as pd

# Read existing HGDP labels from hgdp_1kgp example
print("Reading HGDP labels from existing example...")
hgdp_all_labels = pd.read_csv("${HGDP_METADATA}")
print(f"  Total labels available: {len(hgdp_all_labels)}")

# Read intersected HGDP .fam file to get available samples
fam = pd.read_csv("${TEMP_DIR}/hgdp_intersected.fam", sep=r'\s+', header=None,
                  names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Phenotype'])
available_samples = set(fam['IID'].astype(str))
print(f"  Available in intersected data: {len(available_samples)}")

# Filter labels to only include samples we have
hgdp_all_labels['sample_id'] = hgdp_all_labels['sample_id'].astype(str)
hgdp_labels = hgdp_all_labels[hgdp_all_labels['sample_id'].isin(available_samples)].copy()

# Save filtered labels
hgdp_labels.to_csv("${DATA_DIR}/hgdp_labels.csv", index=False)
print(f"  Saved HGDP labels: {len(hgdp_labels)} samples")

if len(hgdp_labels) < len(available_samples):
    missing = len(available_samples) - len(hgdp_labels)
    print(f"  Warning: {missing} samples in intersected data have no labels")
EOF

# Step 7: Create final processed subsets
print_status "Creating final processed subsets..."

# Create HGDP fit subset (using all intersected HGDP samples)
cp "${TEMP_DIR}/hgdp_intersected.bed" "${DATA_DIR}/fit_subset.bed"
cp "${TEMP_DIR}/hgdp_intersected.bim" "${DATA_DIR}/fit_subset.bim"
cp "${TEMP_DIR}/hgdp_intersected.fam" "${DATA_DIR}/fit_subset.fam"

# Create UKBB project subset
${PLINK2} --bfile ${TEMP_DIR}/ukbb_intersected \
    --keep ${DATA_DIR}/ukbb_samples.txt \
    --make-bed \
    --out ${DATA_DIR}/project_subset

print_success "Final subsets created:"

# Get final sample counts
FIT_SAMPLES=$(wc -l < "${DATA_DIR}/fit_subset.fam")
PROJECT_SAMPLES=$(wc -l < "${DATA_DIR}/project_subset.fam")

echo "  Fit subset (HGDP): $FIT_SAMPLES samples"
echo "  Project subset (UKBB): $PROJECT_SAMPLES samples"
echo "  Common SNPs: $FINAL_SNP_COUNT SNPs"

# Step 8: Cleanup
print_status "Cleaning up temporary files..."
# Keep some key files for inspection, remove others
rm -f "${TEMP_DIR}/ukbb_intersected.*"
rm -f "${TEMP_DIR}/hgdp_intersected.*"
rm -f "${TEMP_DIR}/ukbb_snps.txt"
rm -f "${TEMP_DIR}/hgdp_snps.txt"
rm -f "${TEMP_DIR}/process_ukbb.py"

print_success "Cleanup complete"

# Step 10: Summary
echo ""
echo "========================================="
print_success "Data preparation complete!"
echo "========================================="
echo ""
echo "Generated files:"
echo "  📊 Processed PLINK data:"
echo "    - data/fit_subset.{bed,bim,fam}     (HGDP, $FIT_SAMPLES samples)"
echo "    - data/project_subset.{bed,bim,fam} (UKBB, $PROJECT_SAMPLES samples)"
echo ""
echo "  🏷️ Sample labels:"
echo "    - data/hgdp_labels.csv    (HGDP sample metadata)"
echo "    - data/ukbb_samples.txt   (UKBB sample IDs - $PROJECT_SAMPLES samples)"
echo ""
print_warning "NEXT STEP: Create data/ukbb_project_labels.csv"
echo "  This file must contain labels for the $PROJECT_SAMPLES UKBB samples in ukbb_samples.txt"
echo ""
echo "  Required format (CSV with header):"
echo "    sample_id,self_described_ancestry,Population"
echo "    1234567,British,European"
echo "    2345678,Caribbean,African"
echo "    ..."
echo ""
echo "  Notes:"
echo "    - sample_id must match the IID column (2nd column) from ukbb_samples.txt"
echo "    - Column names and values must match what you define in ukbb_colormap.json"
echo ""
echo "  🎨 Optional: Create visualization colormaps:"
echo "    - data/hgdp_colormap.json (HGDP population colors)"
echo "    - data/ukbb_colormap.json (UKBB ancestry colors)"
echo ""
echo "  📋 Processing logs (in data/temp/):"
echo "    - ukbb_standardized.{bed,bim,fam}, hgdp_standardized.{bed,bim,fam}"
echo "    - common_snps.txt, flip_list.txt, exclude_list.txt"
echo ""
echo "After creating ukbb_project_labels.csv, run the cross-projection pipeline:"
echo "  bash run_pipeline.sh"
echo ""