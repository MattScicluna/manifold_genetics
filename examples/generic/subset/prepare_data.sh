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

# Get script directory and source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "${PROJECT_ROOT}/examples/_shared/preprocessing/common.sh"

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

# Print header
print_header "Generic Subset Data Preparation"

# ============================================================================
# Step 1: Verify Required Files
# ============================================================================
print_status "Verifying required files..."

if ! verify_plink_files "$PLINK"; then
    exit 1
fi
print_success "PLINK files found"

# Check sample list files
if [[ ! -f "$FIT_SAMPLES" ]]; then
    print_error "Fit samples file not found: $FIT_SAMPLES"
    exit 1
fi

if [[ -n "$PROJECT_SAMPLES" ]] && [[ ! -f "$PROJECT_SAMPLES" ]]; then
    print_error "Project samples file not found: $PROJECT_SAMPLES"
    exit 1
fi

print_success "All required files found"

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

if ! find_plink2 "$PROJECT_ROOT"; then
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
    PROJECT_SAMPLES_COUNT=$(get_sample_count "$PLINK")
fi
TOTAL_SAMPLES=$(get_sample_count "$PLINK")
TOTAL_SNPS=$(get_snp_count "$PLINK")

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
FIT_FINAL=$(get_sample_count "${OUTPUT_DIR}/fit_subset")
PROJECT_FINAL=$(get_sample_count "${OUTPUT_DIR}/project_subset")
SUBSET_SNPS=$(get_snp_count "${OUTPUT_DIR}/fit_subset")

print_header "Subset Data Preparation Complete!"

echo "Generated PLINK subsets:"
echo "  fit_subset:"
echo "    - ${OUTPUT_DIR}/fit_subset.{bed,bim,fam}"
echo "    - $FIT_FINAL samples, $SUBSET_SNPS SNPs"
echo ""
echo "  project_subset:"
echo "    - ${OUTPUT_DIR}/project_subset.{bed,bim,fam}"
echo "    - $PROJECT_FINAL samples, $SUBSET_SNPS SNPs"
echo ""
echo "IMPORTANT: Create label CSV files before running the pipeline:"
echo "    - fit_labels.csv: Labels for fit samples"
echo "    - project_labels.csv: Labels for project samples"
echo ""
echo "    Required format:"
echo "      sample_id,Population,Region,..."
echo "      SAMPLE001,British,Europe,..."
echo ""
