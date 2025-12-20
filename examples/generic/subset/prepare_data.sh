#!/bin/bash
#
# Generic Subset Data Preparation Script
#
# This script creates fit and project subsets from a single PLINK dataset.
# Use this when you want to split your own biobank data into training and test sets.
#
# NOTE: This script does NOT create labels. Users must provide pre-created
# label CSV files with columns appropriate for their datasets.
#
# Usage:
#   bash prepare_data.sh \
#       --plink /path/to/data \
#       --fit-samples /path/to/fit_samples.txt \
#       --project-samples /path/to/project_samples.txt \
#       --output-dir ./data
#

set -e

# ============================================================================
# Argument Parsing
# ============================================================================

# Default values
PLINK=""
FIT_SAMPLES=""
PROJECT_SAMPLES=""
OUTPUT_DIR="./data"
MEMORY=100000
THREADS=4

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --plink)
            PLINK="$2"
            shift 2
            ;;
        --fit-samples)
            FIT_SAMPLES="$2"
            shift 2
            ;;
        --project-samples)
            PROJECT_SAMPLES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
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
            echo "Usage: $0 --plink PATH --fit-samples PATH [OPTIONS]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$PLINK" ]] || [[ -z "$FIT_SAMPLES" ]]; then
    echo "Error: Missing required arguments"
    echo ""
    echo "Usage: $0 \\"
    echo "    --plink PATH                # Input PLINK prefix"
    echo "    --fit-samples PATH          # Fit samples list (FID IID format)"
    echo "    [--project-samples PATH]    # Project samples list (optional, uses all if not provided)"
    echo "    [--output-dir PATH]         # Output directory (default: ./data)"
    echo "    [--memory MB]               # plink2 memory limit (default: 100000)"
    echo "    [--threads N]               # Threads (default: 4)"
    echo ""
    echo "NOTE: This script creates PLINK subsets only."
    echo "      Create label CSV files separately before running the pipeline."
    exit 1
fi

# Get script directory for utilities
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
echo "  Generic Subset Data Preparation"
echo "================================================="
echo ""

# ============================================================================
# Step 1: Verify Required Files
# ============================================================================
print_status "Verifying required files..."

REQUIRED_FILES=(
    "${PLINK}.bed"
    "${PLINK}.bim"
    "${PLINK}.fam"
    "${FIT_SAMPLES}"
)

# Add project samples to required files if provided
if [[ -n "$PROJECT_SAMPLES" ]]; then
    REQUIRED_FILES+=("${PROJECT_SAMPLES}")
fi

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
# Step 2: Create Output Directory
# ============================================================================
print_status "Creating output directory..."

mkdir -p "$OUTPUT_DIR"
print_success "Output directory: $OUTPUT_DIR"

# ============================================================================
# Step 3: Find plink2
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
# Step 4: Get Sample Counts
# ============================================================================
print_status "Getting sample counts..."

FIT_SAMPLES_COUNT=$(wc -l < "$FIT_SAMPLES")
if [[ -n "$PROJECT_SAMPLES" ]]; then
    PROJECT_SAMPLES_COUNT=$(wc -l < "$PROJECT_SAMPLES")
else
    PROJECT_SAMPLES_COUNT=$(wc -l < "${PLINK}.fam")
fi
TOTAL_SAMPLES=$(wc -l < "${PLINK}.fam")
TOTAL_SNPS=$(wc -l < "${PLINK}.bim")

echo "  Total samples: $TOTAL_SAMPLES"
echo "  Total SNPs: $TOTAL_SNPS"
echo "  Fit samples: $FIT_SAMPLES_COUNT"
echo "  Project samples: $PROJECT_SAMPLES_COUNT"

# ============================================================================
# Step 5: Create PLINK Subsets
# ============================================================================
print_status "Creating PLINK subsets..."

# Create fit subset
print_status "  Extracting fit subset..."
${PLINK2} --bfile "$PLINK" \
    --keep "$FIT_SAMPLES" \
    --memory $MEMORY \
    --threads $THREADS \
    --make-bed \
    --out "${OUTPUT_DIR}/fit_subset"

print_success "Fit subset created"

# Create project subset
if [[ -n "$PROJECT_SAMPLES" ]]; then
    print_status "  Extracting project subset..."
    ${PLINK2} --bfile "$PLINK" \
        --keep "$PROJECT_SAMPLES" \
        --memory $MEMORY \
        --threads $THREADS \
        --make-bed \
        --out "${OUTPUT_DIR}/project_subset"
    print_success "Project subset created"
else
    print_status "  Using full dataset as project subset..."
    cp "${PLINK}.bed" "${OUTPUT_DIR}/project_subset.bed"
    cp "${PLINK}.bim" "${OUTPUT_DIR}/project_subset.bim"
    cp "${PLINK}.fam" "${OUTPUT_DIR}/project_subset.fam"
    print_success "Project subset created (full dataset)"
fi

# ============================================================================
# Step 6: Summary
# ============================================================================
FIT_FINAL=$(wc -l < "${OUTPUT_DIR}/fit_subset.fam")
PROJECT_FINAL=$(wc -l < "${OUTPUT_DIR}/project_subset.fam")
SUBSET_SNPS=$(wc -l < "${OUTPUT_DIR}/fit_subset.bim")

echo ""
echo "================================================="
print_success "Subset data preparation complete!"
echo "================================================="
echo ""
echo "Generated PLINK subsets:"
echo "  📊 fit_subset:"
echo "    - ${OUTPUT_DIR}/fit_subset.{bed,bim,fam}"
echo "    - $FIT_FINAL samples, $SUBSET_SNPS SNPs"
echo ""
echo "  📊 project_subset:"
echo "    - ${OUTPUT_DIR}/project_subset.{bed,bim,fam}"
echo "    - $PROJECT_FINAL samples, $SUBSET_SNPS SNPs"
echo ""
echo "⚠️  IMPORTANT: Create label CSV files before running the pipeline:"
echo "    - fit_labels.csv: Labels for fit samples"
echo "    - project_labels.csv: Labels for project samples"
echo ""
echo "    Required format:"
echo "      sample_id,Population,Region,..."
echo "      SAMPLE001,British,Europe,..."
echo ""
