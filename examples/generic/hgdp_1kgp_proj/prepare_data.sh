#!/bin/bash
#
# Generic Cross-Projection Data Preparation Script
#
# This script prepares PLINK data for cross-projection analysis by:
# 1. Standardizing SNP IDs between reference and biobank datasets
# 2. Finding SNP intersection
# 3. Checking allele compatibility
# 4. Creating intersected PLINK subsets (fit_subset and project_subset)
#
# NOTE: This script does NOT create labels. Users must provide pre-created
# label CSV files with columns appropriate for their datasets.
#
# Usage:
#   bash prepare_data.sh \
#       --reference-plink /path/to/reference/data \
#       --biobank-plink /path/to/biobank/data \
#       --output-dir ./data
#

set -e

# ============================================================================
# Argument Parsing
# ============================================================================

# Default values
REFERENCE_PLINK=""
BIOBANK_PLINK=""
OUTPUT_DIR="./data"
TEMP_DIR="./data/temp"
MEMORY=100000
THREADS=4

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --reference-plink)
            REFERENCE_PLINK="$2"
            shift 2
            ;;
        --biobank-plink)
            BIOBANK_PLINK="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --temp-dir)
            TEMP_DIR="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --reference-plink PATH --biobank-plink PATH [--output-dir PATH] [--temp-dir PATH] [--memory MB] [--threads N]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$REFERENCE_PLINK" ]] || [[ -z "$BIOBANK_PLINK" ]]; then
    echo "Error: Missing required arguments"
    echo ""
    echo "Usage: $0 \\"
    echo "    --reference-plink PATH       # Reference dataset PLINK prefix"
    echo "    --biobank-plink PATH         # Biobank dataset PLINK prefix"
    echo "    [--output-dir PATH]          # Output directory (default: ./data)"
    echo "    [--temp-dir PATH]            # Temp directory (default: ./data/temp)"
    echo "    [--memory MB]                # plink2 memory limit (default: 100000)"
    echo "    [--threads N]                # Number of threads (default: 4)"
    echo ""
    echo "NOTE: This script creates intersected PLINK files only."
    echo "      Create label CSV files separately before running the pipeline."
    exit 1
fi

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
echo "================================================="
echo "  Generic Cross-Projection Data Preparation"
echo "================================================="
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

# ============================================================================
# Step 1: Verify Input Files
# ============================================================================
print_status "Verifying input files..."

REQUIRED_FILES=(
    "${REFERENCE_PLINK}.bed"
    "${REFERENCE_PLINK}.bim"
    "${REFERENCE_PLINK}.fam"
    "${BIOBANK_PLINK}.bed"
    "${BIOBANK_PLINK}.bim"
    "${BIOBANK_PLINK}.fam"
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

print_success "All required PLINK files found"

# ============================================================================
# Step 2: Find plink2
# ============================================================================
print_status "Looking for plink2..."

PLINK2=""

if [[ -f "${PROJECT_ROOT}/bin/plink2" ]]; then
    PLINK2="${PROJECT_ROOT}/bin/plink2"
    print_success "Using plink2 from bin/plink2"
elif module list 2>&1 | grep -q plink2; then
    PLINK2="plink2"
    print_success "Using plink2 from loaded module"
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

if ! ${PLINK2} --version &> /dev/null; then
    print_error "plink2 found but not executable!"
    exit 1
fi

# ============================================================================
# Step 3: Get Sample Counts
# ============================================================================
print_status "Getting data statistics..."

REF_SAMPLES=$(wc -l < "${REFERENCE_PLINK}.fam")
BIOBANK_SAMPLES=$(wc -l < "${BIOBANK_PLINK}.fam")
REF_SNPS=$(wc -l < "${REFERENCE_PLINK}.bim")
BIOBANK_SNPS=$(wc -l < "${BIOBANK_PLINK}.bim")

echo "  Reference: $REF_SAMPLES samples, $REF_SNPS SNPs"
echo "  Biobank: $BIOBANK_SAMPLES samples, $BIOBANK_SNPS SNPs"

# ============================================================================
# Step 4: Standardize SNP IDs
# ============================================================================
print_status "Standardizing SNP IDs to chr:pos:ref:alt format..."

# Standardize reference dataset
if [[ ! -f "${TEMP_DIR}/reference_standardized.bed" ]]; then
    print_status "  Standardizing reference dataset..."
    ${PLINK2} --bfile ${REFERENCE_PLINK} \
        --set-all-var-ids '@:#:$r:$a' \
        --new-id-max-allele-len 100 missing \
        --memory ${MEMORY} \
        --threads ${THREADS} \
        --make-bed \
        --out ${TEMP_DIR}/reference_standardized
    print_success "  Reference standardized"
else
    print_success "  Reference already standardized (cached)"
fi

# Standardize biobank dataset
if [[ ! -f "${TEMP_DIR}/biobank_standardized.bed" ]]; then
    print_status "  Standardizing biobank dataset..."
    ${PLINK2} --bfile ${BIOBANK_PLINK} \
        --set-all-var-ids '@:#:$r:$a' \
        --new-id-max-allele-len 100 missing \
        --memory ${MEMORY} \
        --threads ${THREADS} \
        --make-bed \
        --out ${TEMP_DIR}/biobank_standardized
    print_success "  Biobank standardized"
else
    print_success "  Biobank already standardized (cached)"
fi

# ============================================================================
# Step 5: Find SNP Intersection
# ============================================================================
print_status "Finding SNP intersection..."

# Extract SNP IDs
awk '{print $2}' ${TEMP_DIR}/reference_standardized.bim | sort > ${TEMP_DIR}/reference_snps.txt
awk '{print $2}' ${TEMP_DIR}/biobank_standardized.bim | sort > ${TEMP_DIR}/biobank_snps.txt

# Find intersection
comm -12 ${TEMP_DIR}/reference_snps.txt ${TEMP_DIR}/biobank_snps.txt > ${TEMP_DIR}/common_snps.txt

COMMON_SNPS=$(wc -l < ${TEMP_DIR}/common_snps.txt)
print_success "Found $COMMON_SNPS common SNPs"

if [[ $COMMON_SNPS -lt 10000 ]]; then
    print_warning "Warning: Very few common SNPs ($COMMON_SNPS). Check SNP ID formats."
fi

# ============================================================================
# Step 6: Check Allele Compatibility
# ============================================================================
print_status "Checking allele compatibility..."

python3 << PYTHON_SCRIPT
import sys

# Read reference and biobank .bim files
ref_bim = {}
with open('${TEMP_DIR}/reference_standardized.bim') as f:
    for line in f:
        fields = line.strip().split('\t')
        snp_id = fields[1]
        ref_allele = fields[4]
        alt_allele = fields[5]
        ref_bim[snp_id] = (ref_allele, alt_allele)

biobank_bim = {}
with open('${TEMP_DIR}/biobank_standardized.bim') as f:
    for line in f:
        fields = line.strip().split('\t')
        snp_id = fields[1]
        ref_allele = fields[4]
        alt_allele = fields[5]
        biobank_bim[snp_id] = (ref_allele, alt_allele)

# Check alleles for common SNPs
common_snps = []
with open('${TEMP_DIR}/common_snps.txt') as f:
    common_snps = [line.strip() for line in f]

flip_snps = []
exclude_snps = []
keep_snps = []

for snp in common_snps:
    if snp not in ref_bim or snp not in biobank_bim:
        continue

    ref_ref, ref_alt = ref_bim[snp]
    bio_ref, bio_alt = biobank_bim[snp]

    # Check if alleles match
    if (ref_ref == bio_ref and ref_alt == bio_alt):
        # Perfect match
        keep_snps.append(snp)
    elif (ref_ref == bio_alt and ref_alt == bio_ref):
        # Flipped - need to flip biobank
        flip_snps.append(snp)
        keep_snps.append(snp)
    else:
        # Incompatible alleles
        exclude_snps.append(snp)

# Write output files
with open('${TEMP_DIR}/snps_to_flip.txt', 'w') as f:
    for snp in flip_snps:
        f.write(f"{snp}\n")

with open('${TEMP_DIR}/snps_to_exclude.txt', 'w') as f:
    for snp in exclude_snps:
        f.write(f"{snp}\n")

with open('${TEMP_DIR}/snps_to_keep.txt', 'w') as f:
    for snp in keep_snps:
        f.write(f"{snp}\n")

print(f"  Allele compatibility check:")
print(f"    Compatible (no flip): {len(keep_snps) - len(flip_snps)}")
print(f"    Compatible (flip):    {len(flip_snps)}")
print(f"    Incompatible:         {len(exclude_snps)}")
print(f"    Final SNPs:           {len(keep_snps)}")
PYTHON_SCRIPT

# ============================================================================
# Step 7: Create Intersected PLINK Files
# ============================================================================
print_status "Creating intersected PLINK subsets..."

# Create reference subset (fit_subset)
print_status "  Creating fit_subset (reference)..."
${PLINK2} --bfile ${TEMP_DIR}/reference_standardized \
    --extract ${TEMP_DIR}/snps_to_keep.txt \
    --memory ${MEMORY} \
    --threads ${THREADS} \
    --make-bed \
    --out ${OUTPUT_DIR}/fit_subset

print_success "  fit_subset created"

# Create biobank subset with flipped alleles (project_subset)
print_status "  Creating project_subset (biobank)..."

FLIP_COUNT=$(wc -l < ${TEMP_DIR}/snps_to_flip.txt)

if [[ $FLIP_COUNT -gt 0 ]]; then
    print_status "  Flipping $FLIP_COUNT SNPs in biobank data..."
    ${PLINK2} --bfile ${TEMP_DIR}/biobank_standardized \
        --extract ${TEMP_DIR}/snps_to_keep.txt \
        --flip ${TEMP_DIR}/snps_to_flip.txt \
        --memory ${MEMORY} \
        --threads ${THREADS} \
        --make-bed \
        --out ${OUTPUT_DIR}/project_subset
else
    ${PLINK2} --bfile ${TEMP_DIR}/biobank_standardized \
        --extract ${TEMP_DIR}/snps_to_keep.txt \
        --memory ${MEMORY} \
        --threads ${THREADS} \
        --make-bed \
        --out ${OUTPUT_DIR}/project_subset
fi

print_success "  project_subset created"

# ============================================================================
# Step 8: Summary
# ============================================================================
FIT_FINAL=$(wc -l < "${OUTPUT_DIR}/fit_subset.fam")
PROJECT_FINAL=$(wc -l < "${OUTPUT_DIR}/project_subset.fam")
FINAL_SNPS=$(wc -l < "${OUTPUT_DIR}/fit_subset.bim")

echo ""
echo "================================================="
print_success "Data preparation complete!"
echo "================================================="
echo ""
echo "Generated intersected PLINK files:"
echo "  📊 fit_subset (reference):"
echo "    - ${OUTPUT_DIR}/fit_subset.{bed,bim,fam}"
echo "    - $FIT_FINAL samples, $FINAL_SNPS SNPs"
echo ""
echo "  📊 project_subset (biobank):"
echo "    - ${OUTPUT_DIR}/project_subset.{bed,bim,fam}"
echo "    - $PROJECT_FINAL samples, $FINAL_SNPS SNPs"
echo ""
echo "⚠️  IMPORTANT: Create label CSV files before running the pipeline:"
echo "    - fit_labels.csv: Labels for reference samples"
echo "    - project_labels.csv: Labels for biobank samples"
echo ""
echo "    Required format:"
echo "      sample_id,Population,Region,..."
echo "      SAMPLE001,Yoruba,Africa,..."
echo ""
