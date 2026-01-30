#!/bin/bash
#
# Generic Cross-Projection Data Preparation Script
#
# This script prepares PLINK data for cross-projection analysis by calling
# the shared preprocessing script. It performs:
#   - Indel removal
#   - Genotype missingness filtering
#   - SNP ID standardization
#   - SNP intersection and allele compatibility check
#   - Creation of intersected PLINK files
#
# By default, this runs "lite" preprocessing (no WRayner, GIAB, HLA, LD pruning, MAF).
# For full preprocessing, remove the --skip-* flags below.
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

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SHARED_PREPROCESS="${PROJECT_ROOT}/examples/_shared/preprocessing/preprocess_cross_projection.sh"

# Source common utilities
source "${PROJECT_ROOT}/examples/_shared/preprocessing/common.sh"

# ============================================================================
# Argument Parsing
# ============================================================================

# Default values
REFERENCE_PLINK=""
BIOBANK_PLINK=""
OUTPUT_DIR="./data"
TEMP_DIR=""
MEMORY=100000
THREADS="${SLURM_CPUS_PER_TASK:-4}"

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
        -h|--help)
            echo "Generic Cross-Projection Data Preparation"
            echo ""
            echo "Usage: $0 --reference-plink PATH --biobank-plink PATH [OPTIONS]"
            echo ""
            echo "Required arguments:"
            echo "  --reference-plink PATH    Reference dataset PLINK prefix"
            echo "  --biobank-plink PATH      Biobank dataset PLINK prefix"
            echo ""
            echo "Optional arguments:"
            echo "  --output-dir PATH         Output directory (default: ./data)"
            echo "  --temp-dir PATH           Temp directory (default: OUTPUT_DIR/temp)"
            echo "  --memory MB               plink2 memory limit (default: 100000)"
            echo "  --threads N               Number of threads (default: 4)"
            echo ""
            echo "NOTE: This script creates intersected PLINK files only."
            echo "      Create label CSV files separately before running the pipeline."
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$REFERENCE_PLINK" ]] || [[ -z "$BIOBANK_PLINK" ]]; then
    print_error "Missing required arguments"
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

# Set default temp dir if not provided
TEMP_DIR="${TEMP_DIR:-${OUTPUT_DIR}/temp}"

# Create directories
mkdir -p "$OUTPUT_DIR" "$TEMP_DIR"

# Print header
print_header "Generic Cross-Projection Data Preparation"

echo "Configuration:"
echo "  Reference: ${REFERENCE_PLINK}"
echo "  Biobank: ${BIOBANK_PLINK}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Threads: ${THREADS}"
echo ""

# ============================================================================
# Call Shared Preprocessing Script
# ============================================================================
print_status "Calling shared preprocessing script (lite mode)..."
echo ""

# Run lite preprocessing (skip expensive steps)
# Remove --skip-* flags for full preprocessing
bash "${SHARED_PREPROCESS}" \
    --reference-plink "$REFERENCE_PLINK" \
    --biobank-plink "$BIOBANK_PLINK" \
    --output-dir "$OUTPUT_DIR" \
    --temp-dir "$TEMP_DIR" \
    --memory "$MEMORY" \
    --threads "$THREADS" \
    --skip-wrayner \
    --skip-giab \
    --skip-hla \
    --skip-ld-prune \
    --skip-dedup \
    --skip-maf

# ============================================================================
# Summary
# ============================================================================
FIT_SAMPLES=$(get_sample_count "${OUTPUT_DIR}/fit_subset")
PROJECT_SAMPLES=$(get_sample_count "${OUTPUT_DIR}/project_subset")
FINAL_SNPS=$(get_snp_count "${OUTPUT_DIR}/fit_subset")

print_header "Data Preparation Complete!"

echo "Generated intersected PLINK files:"
echo "  fit_subset (reference):"
echo "    - ${OUTPUT_DIR}/fit_subset.{bed,bim,fam}"
echo "    - $FIT_SAMPLES samples, $FINAL_SNPS SNPs"
echo ""
echo "  project_subset (biobank):"
echo "    - ${OUTPUT_DIR}/project_subset.{bed,bim,fam}"
echo "    - $PROJECT_SAMPLES samples, $FINAL_SNPS SNPs"
echo ""
echo "IMPORTANT: Create label CSV files before running the pipeline:"
echo "    - fit_labels.csv: Labels for reference samples"
echo "    - project_labels.csv: Labels for biobank samples"
echo ""
echo "    Required format:"
echo "      sample_id,Population,Region,..."
echo "      SAMPLE001,Yoruba,Africa,..."
echo ""
