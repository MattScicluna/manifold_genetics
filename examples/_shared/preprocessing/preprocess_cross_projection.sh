#!/bin/bash
#
# Cross-Projection Preprocessing Script
#
# This script performs comprehensive preprocessing for cross-cohort projection:
#   - Filter both datasets (indels, MAF, geno)
#   - Optional: Deduplicate reference (remove multi-allelic positions)
#   - Optional: Exclude GIAB difficult regions
#   - Optional: Exclude HLA/MHC region
#   - Optional: WRayner harmonization (check against TOPMed reference)
#   - Optional: LD pruning
#   - SNP intersection and allele compatibility check
#   - Create final intersected PLINK files
#
# Usage:
#   bash preprocess_cross_projection.sh \
#       --reference-plink /path/to/reference \
#       --biobank-plink /path/to/biobank \
#       --output-dir /path/to/output \
#       [OPTIONS]
#
# For a "lite" version without external tools:
#   bash preprocess_cross_projection.sh \
#       --reference-plink /path/to/reference \
#       --biobank-plink /path/to/biobank \
#       --output-dir /path/to/output \
#       --skip-wrayner --skip-giab --skip-hla --skip-ld-prune --skip-dedup
#

set -e

# Get script directory and source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# ============================================================================
# Default Values
# ============================================================================
REFERENCE_PLINK=""
BIOBANK_PLINK=""
OUTPUT_DIR=""
TEMP_DIR=""
TOOLS_DIR="${SCRIPT_DIR}/../tools"

# Performance
MEMORY="${MEMORY:-100000}"
THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Filtering parameters
MAF_THRESHOLD="${MAF_THRESHOLD:-0.01}"
GENO_THRESHOLD="${GENO_THRESHOLD:-0.05}"

# LD pruning parameters
LD_WINDOW="${LD_WINDOW:-150}"
LD_STEP="${LD_STEP:-1}"
LD_R2="${LD_R2:-0.05}"

# Skip flags (all default to false = steps enabled)
SKIP_WRAYNER=false
SKIP_GIAB=false
SKIP_HLA=false
SKIP_LD_PRUNE=false
SKIP_DEDUP=false

# Reference-specific options
REFERENCE_HAS_CHR_PREFIX=false

# ============================================================================
# Argument Parsing
# ============================================================================
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
        --tools-dir)
            TOOLS_DIR="$2"
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
        --maf)
            MAF_THRESHOLD="$2"
            shift 2
            ;;
        --geno)
            GENO_THRESHOLD="$2"
            shift 2
            ;;
        --ld-window)
            LD_WINDOW="$2"
            shift 2
            ;;
        --ld-step)
            LD_STEP="$2"
            shift 2
            ;;
        --ld-r2)
            LD_R2="$2"
            shift 2
            ;;
        --skip-wrayner)
            SKIP_WRAYNER=true
            shift
            ;;
        --skip-giab)
            SKIP_GIAB=true
            shift
            ;;
        --skip-hla)
            SKIP_HLA=true
            shift
            ;;
        --skip-ld-prune)
            SKIP_LD_PRUNE=true
            shift
            ;;
        --skip-dedup)
            SKIP_DEDUP=true
            shift
            ;;
        --reference-has-chr-prefix)
            REFERENCE_HAS_CHR_PREFIX=true
            shift
            ;;
        -h|--help)
            echo "Cross-Projection Preprocessing Script"
            echo ""
            echo "Usage: $0 --reference-plink PATH --biobank-plink PATH --output-dir PATH [OPTIONS]"
            echo ""
            echo "Required arguments:"
            echo "  --reference-plink PATH    Reference dataset PLINK prefix"
            echo "  --biobank-plink PATH      Biobank dataset PLINK prefix"
            echo "  --output-dir PATH         Output directory for final files"
            echo ""
            echo "Optional arguments:"
            echo "  --temp-dir PATH           Temp directory (default: OUTPUT_DIR/temp)"
            echo "  --tools-dir PATH          Directory for external tools (default: _shared/tools)"
            echo "  --memory MB               plink2 memory limit (default: 100000)"
            echo "  --threads N               Number of threads (default: SLURM_CPUS_PER_TASK or 4)"
            echo ""
            echo "Filtering parameters:"
            echo "  --maf FLOAT               MAF threshold (default: 0.01)"
            echo "  --geno FLOAT              Genotype missingness threshold (default: 0.05)"
            echo ""
            echo "LD pruning parameters:"
            echo "  --ld-window INT           LD pruning window in kb (default: 150)"
            echo "  --ld-step INT             LD pruning step size (default: 1)"
            echo "  --ld-r2 FLOAT             LD pruning r² threshold (default: 0.05)"
            echo ""
            echo "Skip flags (for lite preprocessing):"
            echo "  --skip-wrayner            Skip WRayner harmonization"
            echo "  --skip-giab               Skip GIAB difficult regions exclusion"
            echo "  --skip-hla                Skip HLA/MHC region exclusion"
            echo "  --skip-ld-prune           Skip LD pruning"
            echo "  --skip-dedup              Skip deduplication of reference"
            echo ""
            echo "Reference-specific options:"
            echo "  --reference-has-chr-prefix   Reference already has 'chr' prefix in chromosomes"
            echo ""
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
if [[ -z "$REFERENCE_PLINK" ]] || [[ -z "$BIOBANK_PLINK" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    print_error "Missing required arguments"
    echo ""
    echo "Usage: $0 --reference-plink PATH --biobank-plink PATH --output-dir PATH [OPTIONS]"
    echo "Use --help for full usage information"
    exit 1
fi

# Set default temp dir if not provided
TEMP_DIR="${TEMP_DIR:-${OUTPUT_DIR}/temp}"

# Create directories
mkdir -p "$OUTPUT_DIR" "$TEMP_DIR" "$TOOLS_DIR"

# Get project root for Python utilities
PROJECT_ROOT="$(get_project_root)"

# ============================================================================
# Header
# ============================================================================
print_header "Cross-Projection Preprocessing"

echo "Configuration:"
echo "  Reference: ${REFERENCE_PLINK}"
echo "  Biobank: ${BIOBANK_PLINK}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Temp: ${TEMP_DIR}"
echo "  Threads: ${THREADS}"
echo ""
echo "Filtering parameters:"
echo "  MAF threshold: ${MAF_THRESHOLD}"
echo "  Geno threshold: ${GENO_THRESHOLD}"
if [[ "$SKIP_LD_PRUNE" != "true" ]]; then
    echo "  LD pruning: ${LD_WINDOW}kb window, step ${LD_STEP}, r²=${LD_R2}"
fi
echo ""
echo "Steps enabled:"
echo "  Filtering: yes"
echo "  Deduplication: $([ "$SKIP_DEDUP" == "true" ] && echo "no" || echo "yes")"
echo "  GIAB exclusion: $([ "$SKIP_GIAB" == "true" ] && echo "no" || echo "yes")"
echo "  HLA exclusion: $([ "$SKIP_HLA" == "true" ] && echo "no" || echo "yes")"
echo "  WRayner harmonization: $([ "$SKIP_WRAYNER" == "true" ] && echo "no" || echo "yes")"
echo "  LD pruning: $([ "$SKIP_LD_PRUNE" == "true" ] && echo "no" || echo "yes")"
echo ""

# ============================================================================
# Step 0: Find Required Tools
# ============================================================================
print_header "Checking Required Tools"

print_status "Looking for plink2..."
if ! find_plink2 "$PROJECT_ROOT"; then
    exit 1
fi

print_status "Looking for plink (v1.9)..."
if ! find_plink "$PROJECT_ROOT"; then
    exit 1
fi

# ============================================================================
# Step 1: Verify Input Files
# ============================================================================
print_header "Verifying Input Files"

print_status "Checking reference dataset..."
if ! verify_plink_files "$REFERENCE_PLINK"; then
    exit 1
fi
print_plink_stats "$REFERENCE_PLINK" "Reference"

print_status "Checking biobank dataset..."
if ! verify_plink_files "$BIOBANK_PLINK"; then
    exit 1
fi
print_plink_stats "$BIOBANK_PLINK" "Biobank"

# ============================================================================
# Step 2: Filter Biobank Data
# ============================================================================
print_header "Filtering Biobank Data"

BIOBANK_FILTERED="${TEMP_DIR}/biobank_filtered"

if [[ ! -f "${BIOBANK_FILTERED}.bed" ]]; then
    print_status "Filtering biobank (MAF > ${MAF_THRESHOLD}, missing < ${GENO_THRESHOLD}, no indels)..."

    INITIAL_BIOBANK_SNPS=$(get_snp_count "$BIOBANK_PLINK")
    echo "  Initial biobank SNPs: $INITIAL_BIOBANK_SNPS"

    # Step 1: Remove indels (keep only single-nucleotide biallelic SNPs)
    print_status "  Removing indels (keeping only A/T/G/C SNPs)..."
    awk 'length($5) == 1 && length($6) == 1 && $5 ~ /^[ATGC]$/ && $6 ~ /^[ATGC]$/ {print $2}' \
        "${BIOBANK_PLINK}.bim" > "${TEMP_DIR}/biobank_snps_no_indels.txt"
    NO_INDEL_SNPS=$(wc -l < "${TEMP_DIR}/biobank_snps_no_indels.txt")
    echo "    SNPs after indel removal: $NO_INDEL_SNPS"

    # Step 2: Apply quality filters
    ${PLINK2} --bfile ${BIOBANK_PLINK} \
        --extract ${TEMP_DIR}/biobank_snps_no_indels.txt \
        --geno ${GENO_THRESHOLD} \
        --maf ${MAF_THRESHOLD} \
        --output-chr chrM \
        --memory ${MEMORY} \
        --threads ${THREADS} \
        --make-bed \
        --out ${BIOBANK_FILTERED}

    BIOBANK_FILTERED_SNPS=$(get_snp_count "$BIOBANK_FILTERED")
    print_success "Biobank SNPs after filtering: ${BIOBANK_FILTERED_SNPS}"
else
    print_success "Using existing filtered biobank data"
    BIOBANK_FILTERED_SNPS=$(get_snp_count "$BIOBANK_FILTERED")
    echo "  Filtered biobank SNPs: ${BIOBANK_FILTERED_SNPS}"
fi

# ============================================================================
# Step 3: Filter Reference Data
# ============================================================================
print_header "Filtering Reference Data"

REFERENCE_FILTERED="${TEMP_DIR}/reference_filtered"
REFERENCE_INPUT="$REFERENCE_PLINK"

if [[ ! -f "${REFERENCE_FILTERED}.bed" ]]; then
    INITIAL_REF_SNPS=$(get_snp_count "$REFERENCE_INPUT")
    echo "  Initial reference SNPs: $INITIAL_REF_SNPS"

    # -------------------------------------------------------------------------
    # 3a. Remove indels
    # -------------------------------------------------------------------------
    print_status "[3a] Removing indels (keeping only A/T/G/C SNPs)..."
    awk 'length($5) == 1 && length($6) == 1 && $5 ~ /^[ATGC]$/ && $6 ~ /^[ATGC]$/ {print $2}' \
        "${REFERENCE_INPUT}.bim" > "${TEMP_DIR}/reference_snps_no_indels.txt"
    NO_INDEL_COUNT=$(wc -l < "${TEMP_DIR}/reference_snps_no_indels.txt")
    echo "    SNPs after indel removal: $NO_INDEL_COUNT"

    # Create intermediate file without indels
    REFERENCE_NO_INDELS="${TEMP_DIR}/reference_no_indels"
    ${PLINK} --bfile ${REFERENCE_INPUT} \
        --extract ${TEMP_DIR}/reference_snps_no_indels.txt \
        --keep-allele-order \
        --make-bed \
        --out ${REFERENCE_NO_INDELS}

    CURRENT_REF="${REFERENCE_NO_INDELS}"

    # -------------------------------------------------------------------------
    # 3b. Remove duplicate/multi-allelic positions (optional)
    # -------------------------------------------------------------------------
    if [[ "$SKIP_DEDUP" != "true" ]]; then
        print_status "[3b] Removing duplicate/multi-allelic positions..."

        # Calculate missingness
        echo "    Calculating per-SNP missingness..."
        ${PLINK} --bfile ${CURRENT_REF} \
            --missing \
            --out ${TEMP_DIR}/reference_missing

        # Calculate allele frequencies
        echo "    Calculating allele frequencies..."
        ${PLINK} --bfile ${CURRENT_REF} \
            --freq \
            --out ${TEMP_DIR}/reference_freq

        # Run Python script to identify SNPs to keep
        echo "    Identifying best SNP per position..."
        python3 -m manifold_genetics.utils.filter_duplicates \
            --bim ${CURRENT_REF}.bim \
            --lmiss ${TEMP_DIR}/reference_missing.lmiss \
            --frq ${TEMP_DIR}/reference_freq.frq \
            --out ${TEMP_DIR}/reference_dedup

        DEDUP_COUNT=$(wc -l < "${TEMP_DIR}/reference_dedup.keep_snps.txt")
        echo "    SNPs after deduplication: $DEDUP_COUNT"

        # Create intermediate file without duplicates
        REFERENCE_DEDUP="${TEMP_DIR}/reference_deduped"
        ${PLINK} --bfile ${CURRENT_REF} \
            --extract ${TEMP_DIR}/reference_dedup.keep_snps.txt \
            --keep-allele-order \
            --make-bed \
            --out ${REFERENCE_DEDUP}

        CURRENT_REF="${REFERENCE_DEDUP}"
    else
        print_status "[3b] Skipping deduplication (--skip-dedup)"
    fi

    # -------------------------------------------------------------------------
    # 3c. Exclude GIAB difficult regions (optional)
    # -------------------------------------------------------------------------
    if [[ "$SKIP_GIAB" != "true" ]]; then
        print_status "[3c] Downloading GIAB difficult regions blacklist..."

        GIAB_URL="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/genome-stratifications/v3.6/GRCh38@all/Union/GRCh38_alldifficultregions.bed.gz"
        GIAB_BED="${TOOLS_DIR}/giab/GRCh38_alldifficultregions.bed"
        mkdir -p "${TOOLS_DIR}/giab"

        if [[ ! -f "${GIAB_BED}" ]]; then
            echo "    Downloading from NCBI..."
            curl -sL "${GIAB_URL}" | gunzip > "${GIAB_BED}"
            echo "    Downloaded $(wc -l < "${GIAB_BED}") regions"
        else
            echo "    GIAB blacklist already downloaded"
        fi

        EXCLUDE_BED="${TEMP_DIR}/exclude_regions.bed"
        cp "${GIAB_BED}" "${EXCLUDE_BED}"
    else
        print_status "[3c] Skipping GIAB exclusion (--skip-giab)"
        EXCLUDE_BED="${TEMP_DIR}/exclude_regions.bed"
        > "${EXCLUDE_BED}"  # Create empty file
    fi

    # -------------------------------------------------------------------------
    # 3d. Exclude HLA/MHC region (optional)
    # -------------------------------------------------------------------------
    if [[ "$SKIP_HLA" != "true" ]]; then
        print_status "[3d] Adding HLA/MHC exclusion region..."

        HLA_BED="${TEMP_DIR}/HLA_region.bed"
        # HLA region: chr6:28.5-33.5Mb
        if [[ "$REFERENCE_HAS_CHR_PREFIX" == "true" ]]; then
            echo -e "chr6\t28510120\t33480577" > "${HLA_BED}"
        else
            echo -e "6\t28510120\t33480577" > "${HLA_BED}"
        fi

        # Append HLA to exclusion regions
        cat "${HLA_BED}" >> "${EXCLUDE_BED}"
        echo "    Added HLA/MHC region to exclusion list"
    else
        print_status "[3d] Skipping HLA exclusion (--skip-hla)"
    fi

    # Apply region exclusions if any
    if [[ -s "${EXCLUDE_BED}" ]]; then
        EXCLUDE_REGION_COUNT=$(wc -l < "${EXCLUDE_BED}")
        print_status "Excluding ${EXCLUDE_REGION_COUNT} difficult/HLA regions..."

        REFERENCE_NO_DIFFICULT="${TEMP_DIR}/reference_no_difficult"
        ${PLINK2} --bfile ${CURRENT_REF} \
            --exclude range ${EXCLUDE_BED} \
            --make-bed \
            --out ${REFERENCE_NO_DIFFICULT}

        NO_DIFFICULT_COUNT=$(get_snp_count "$REFERENCE_NO_DIFFICULT")
        echo "    SNPs after excluding difficult regions: $NO_DIFFICULT_COUNT"

        CURRENT_REF="${REFERENCE_NO_DIFFICULT}"
    fi

    # -------------------------------------------------------------------------
    # 3e. Apply MAF and missingness filters
    # -------------------------------------------------------------------------
    print_status "[3e] Applying MAF (>= ${MAF_THRESHOLD}) and geno (< ${GENO_THRESHOLD}) filters..."

    ${PLINK2} --bfile ${CURRENT_REF} \
        --maf ${MAF_THRESHOLD} \
        --geno ${GENO_THRESHOLD} \
        --output-chr chrM \
        --make-bed \
        --out ${REFERENCE_FILTERED}

    REF_FILTERED_SNPS=$(get_snp_count "$REFERENCE_FILTERED")
    print_success "Reference SNPs after filtering: ${REF_FILTERED_SNPS}"
else
    print_success "Using existing filtered reference data"
    REF_FILTERED_SNPS=$(get_snp_count "$REFERENCE_FILTERED")
    echo "  Filtered reference SNPs: ${REF_FILTERED_SNPS}"
fi

# ============================================================================
# Step 4: Check Overlap After Filtering
# ============================================================================
print_header "Checking SNP Overlap After Filtering"

awk '{print $1":"$4}' "${BIOBANK_FILTERED}.bim" | sort > "${TEMP_DIR}/biobank_filtered_pos.txt"
awk '{print $1":"$4}' "${REFERENCE_FILTERED}.bim" | sort > "${TEMP_DIR}/reference_filtered_pos.txt"
OVERLAP_AFTER=$(comm -12 "${TEMP_DIR}/biobank_filtered_pos.txt" "${TEMP_DIR}/reference_filtered_pos.txt" | wc -l)
print_success "Overlap after filtering: ${OVERLAP_AFTER} positions"

# ============================================================================
# Step 5: WRayner Harmonization (optional)
# ============================================================================
if [[ "$SKIP_WRAYNER" != "true" ]]; then
    print_header "WRayner Harmonization"

    echo "This step uses WRayner's HRC-1000G-check-bim tool to:"
    echo "  - Check SNPs against TOPMed reference"
    echo "  - Identify and exclude problematic positions"
    echo "  - Standardize SNP IDs to chr:pos:ref:alt format"
    echo ""

    WRAYNER_DIR="${TOOLS_DIR}/wrayner"
    mkdir -p "${WRAYNER_DIR}"

    # -------------------------------------------------------------------------
    # 5a. Download WRayner tools if needed
    # -------------------------------------------------------------------------
    WRAYNER_SCRIPT="${WRAYNER_DIR}/HRC-1000G-check-bim.pl"
    if [[ ! -f "${WRAYNER_SCRIPT}" ]]; then
        print_status "[5a] Downloading WRayner HRC-1000G-check-bim tools..."
        curl -sL "https://www.chg.ox.ac.uk/~wrayner/tools/HRC-1000G-check-bim-v4.3.0.zip" -o "${WRAYNER_DIR}/HRC-1000G-check-bim.zip"
        unzip -q -o "${WRAYNER_DIR}/HRC-1000G-check-bim.zip" -d "${WRAYNER_DIR}/"
        print_success "WRayner tools downloaded"
    else
        print_success "[5a] WRayner tools already downloaded"
    fi

    # -------------------------------------------------------------------------
    # 5b. Download TOPMed reference file if needed
    # -------------------------------------------------------------------------
    TOPMED_REF="${TOOLS_DIR}/topmed/bravo-dbsnp-all.hrc_format.tab.gz"
    TOPMED_REF_URL="https://www.dropbox.com/scl/fi/jiy6ty8pmrrr2s5eox1nf/bravo-dbsnp-all.hrc_format.tab.gz?rlkey=aihwmah4uwvazla7yl438odut&st=8gklc135&dl=1"
    mkdir -p "${TOOLS_DIR}/topmed"

    if [[ ! -f "${TOPMED_REF}" ]]; then
        print_status "[5b] Downloading TOPMed reference file..."
        echo "    This is a large file (~2GB), please wait..."
        curl -L -o "${TOPMED_REF}" "${TOPMED_REF_URL}"
        if [[ ! -f "${TOPMED_REF}" ]]; then
            print_error "Failed to download TOPMed reference file"
            echo "  Please manually download from: ${TOPMED_REF_URL}"
            echo "  And place at: ${TOPMED_REF}"
            exit 1
        fi
        print_success "TOPMed reference downloaded"
    else
        print_success "[5b] TOPMed reference already downloaded"
    fi

    # -------------------------------------------------------------------------
    # 5c. Harmonize biobank dataset
    # -------------------------------------------------------------------------
    print_status "[5c] Harmonizing biobank dataset..."

    BIOBANK_HARMONIZED="${TEMP_DIR}/biobank_harmonized"
    BIOBANK_WRAYNER_DIR="${TEMP_DIR}/biobank_wrayner"
    mkdir -p "${BIOBANK_WRAYNER_DIR}"

    if [[ ! -f "${BIOBANK_HARMONIZED}.bed" ]]; then
        # Create temporary BIM without "chr" prefix (WRayner expects numeric chromosomes)
        echo "    Creating temporary BIM without chr prefix for WRayner..."
        sed 's/^chr//' "${BIOBANK_FILTERED}.bim" > "${BIOBANK_WRAYNER_DIR}/biobank_nochr.bim"

        # Compute allele frequencies
        echo "    Computing allele frequencies..."
        ${PLINK} --bfile ${BIOBANK_FILTERED} \
            --freq \
            --out ${BIOBANK_WRAYNER_DIR}/biobank_freq

        # Run WRayner tool
        echo "    Running WRayner HRC check..."
        ORIGINAL_DIR=$(pwd)
        cd "${BIOBANK_WRAYNER_DIR}"
        perl "${WRAYNER_SCRIPT}" \
            -b "${BIOBANK_WRAYNER_DIR}/biobank_nochr.bim" \
            -f "${BIOBANK_WRAYNER_DIR}/biobank_freq.frq" \
            -r "${TOPMED_REF}" \
            -h
        cd "${ORIGINAL_DIR}"

        # Move WRayner temp files
        mkdir -p "${BIOBANK_WRAYNER_DIR}/wrayner_tmp"
        mv ${BIOBANK_WRAYNER_DIR}/Strand-Flip* ${BIOBANK_WRAYNER_DIR}/Position* ${BIOBANK_WRAYNER_DIR}/Chromosome* \
           ${BIOBANK_WRAYNER_DIR}/Run-plink.sh ${BIOBANK_WRAYNER_DIR}/LOG* ${BIOBANK_WRAYNER_DIR}/ID* \
           ${BIOBANK_WRAYNER_DIR}/FreqPlot* ${BIOBANK_WRAYNER_DIR}/Force-Allele1* \
           ${BIOBANK_WRAYNER_DIR}/wrayner_tmp/ 2>/dev/null || true

        # Find the exclude file
        EXCLUDE_FILE=$(ls ${BIOBANK_WRAYNER_DIR}/Exclude-*.txt 2>/dev/null | head -1)
        if [[ -n "${EXCLUDE_FILE}" ]] && [[ -f "${EXCLUDE_FILE}" ]]; then
            EXCLUDE_COUNT=$(wc -l < "${EXCLUDE_FILE}")
            echo "    Excluding ${EXCLUDE_COUNT} problematic SNPs..."
            mv "${EXCLUDE_FILE}" "${BIOBANK_WRAYNER_DIR}/wrayner_tmp/"
            EXCLUDE_FILE="${BIOBANK_WRAYNER_DIR}/wrayner_tmp/$(basename ${EXCLUDE_FILE})"

            ${PLINK} --bfile ${BIOBANK_FILTERED} \
                --exclude "${EXCLUDE_FILE}" \
                --make-bed \
                --out ${TEMP_DIR}/biobank_wrayner_filtered
        else
            echo "    No exclude file generated, using filtered data directly..."
            cp ${BIOBANK_FILTERED}.bed ${TEMP_DIR}/biobank_wrayner_filtered.bed
            cp ${BIOBANK_FILTERED}.bim ${TEMP_DIR}/biobank_wrayner_filtered.bim
            cp ${BIOBANK_FILTERED}.fam ${TEMP_DIR}/biobank_wrayner_filtered.fam
        fi

        # Rename SNP IDs to chr:pos:ref:alt format
        echo "    Standardizing SNP IDs to chr:pos:ref:alt format..."
        cp ${TEMP_DIR}/biobank_wrayner_filtered.bim ${TEMP_DIR}/biobank_wrayner_filtered.bim.bkp
        awk '{print $1"\tchr"$1":"$4":"$6":"$5"\t"$3"\t"$4"\t"$5"\t"$6}' \
            ${TEMP_DIR}/biobank_wrayner_filtered.bim.bkp > ${TEMP_DIR}/biobank_wrayner_filtered.bim

        # Create final harmonized output
        cp ${TEMP_DIR}/biobank_wrayner_filtered.bed ${BIOBANK_HARMONIZED}.bed
        cp ${TEMP_DIR}/biobank_wrayner_filtered.bim ${BIOBANK_HARMONIZED}.bim
        cp ${TEMP_DIR}/biobank_wrayner_filtered.fam ${BIOBANK_HARMONIZED}.fam

        BIOBANK_HARMONIZED_SNPS=$(get_snp_count "$BIOBANK_HARMONIZED")
        print_success "Biobank harmonized: ${BIOBANK_HARMONIZED_SNPS} SNPs"
    else
        print_success "Biobank dataset already harmonized"
        BIOBANK_HARMONIZED_SNPS=$(get_snp_count "$BIOBANK_HARMONIZED")
        echo "    Harmonized biobank SNPs: ${BIOBANK_HARMONIZED_SNPS}"
    fi

    # -------------------------------------------------------------------------
    # 5d. Harmonize reference dataset
    # -------------------------------------------------------------------------
    print_status "[5d] Harmonizing reference dataset..."

    REFERENCE_HARMONIZED="${TEMP_DIR}/reference_harmonized"
    REFERENCE_WRAYNER_DIR="${TEMP_DIR}/reference_wrayner"
    mkdir -p "${REFERENCE_WRAYNER_DIR}"

    if [[ ! -f "${REFERENCE_HARMONIZED}.bed" ]]; then
        # Create temporary BIM without "chr" prefix
        echo "    Creating temporary BIM without chr prefix for WRayner..."
        sed 's/^chr//' "${REFERENCE_FILTERED}.bim" > "${REFERENCE_WRAYNER_DIR}/reference_nochr.bim"

        # Compute allele frequencies
        echo "    Computing allele frequencies..."
        ${PLINK} --bfile ${REFERENCE_FILTERED} \
            --freq \
            --out ${REFERENCE_WRAYNER_DIR}/reference_freq

        # Run WRayner tool
        echo "    Running WRayner HRC check..."
        ORIGINAL_DIR=$(pwd)
        cd "${REFERENCE_WRAYNER_DIR}"
        perl "${WRAYNER_SCRIPT}" \
            -b "${REFERENCE_WRAYNER_DIR}/reference_nochr.bim" \
            -f "${REFERENCE_WRAYNER_DIR}/reference_freq.frq" \
            -r "${TOPMED_REF}" \
            -h
        cd "${ORIGINAL_DIR}"

        # Move WRayner temp files
        mkdir -p "${REFERENCE_WRAYNER_DIR}/wrayner_tmp"
        mv ${REFERENCE_WRAYNER_DIR}/Strand-Flip* ${REFERENCE_WRAYNER_DIR}/Position* ${REFERENCE_WRAYNER_DIR}/Chromosome* \
           ${REFERENCE_WRAYNER_DIR}/Run-plink.sh ${REFERENCE_WRAYNER_DIR}/LOG* ${REFERENCE_WRAYNER_DIR}/ID* \
           ${REFERENCE_WRAYNER_DIR}/FreqPlot* ${REFERENCE_WRAYNER_DIR}/Force-Allele1* \
           ${REFERENCE_WRAYNER_DIR}/wrayner_tmp/ 2>/dev/null || true

        # Find the exclude file
        EXCLUDE_FILE=$(ls ${REFERENCE_WRAYNER_DIR}/Exclude-*.txt 2>/dev/null | head -1)
        if [[ -n "${EXCLUDE_FILE}" ]] && [[ -f "${EXCLUDE_FILE}" ]]; then
            EXCLUDE_COUNT=$(wc -l < "${EXCLUDE_FILE}")
            echo "    Excluding ${EXCLUDE_COUNT} problematic SNPs..."
            mv "${EXCLUDE_FILE}" "${REFERENCE_WRAYNER_DIR}/wrayner_tmp/"
            EXCLUDE_FILE="${REFERENCE_WRAYNER_DIR}/wrayner_tmp/$(basename ${EXCLUDE_FILE})"

            ${PLINK} --bfile ${REFERENCE_FILTERED} \
                --exclude "${EXCLUDE_FILE}" \
                --make-bed \
                --out ${TEMP_DIR}/reference_wrayner_filtered
        else
            echo "    No exclude file generated, using filtered data directly..."
            cp ${REFERENCE_FILTERED}.bed ${TEMP_DIR}/reference_wrayner_filtered.bed
            cp ${REFERENCE_FILTERED}.bim ${TEMP_DIR}/reference_wrayner_filtered.bim
            cp ${REFERENCE_FILTERED}.fam ${TEMP_DIR}/reference_wrayner_filtered.fam
        fi

        # Rename SNP IDs to chr:pos:ref:alt format
        echo "    Standardizing SNP IDs to chr:pos:ref:alt format..."
        cp ${TEMP_DIR}/reference_wrayner_filtered.bim ${TEMP_DIR}/reference_wrayner_filtered.bim.bkp
        awk '{print $1"\tchr"$1":"$4":"$6":"$5"\t"$3"\t"$4"\t"$5"\t"$6}' \
            ${TEMP_DIR}/reference_wrayner_filtered.bim.bkp > ${TEMP_DIR}/reference_wrayner_filtered.bim

        # Create final harmonized output
        cp ${TEMP_DIR}/reference_wrayner_filtered.bed ${REFERENCE_HARMONIZED}.bed
        cp ${TEMP_DIR}/reference_wrayner_filtered.bim ${REFERENCE_HARMONIZED}.bim
        cp ${TEMP_DIR}/reference_wrayner_filtered.fam ${REFERENCE_HARMONIZED}.fam

        REFERENCE_HARMONIZED_SNPS=$(get_snp_count "$REFERENCE_HARMONIZED")
        print_success "Reference harmonized: ${REFERENCE_HARMONIZED_SNPS} SNPs"
    else
        print_success "Reference dataset already harmonized"
        REFERENCE_HARMONIZED_SNPS=$(get_snp_count "$REFERENCE_HARMONIZED")
        echo "    Harmonized reference SNPs: ${REFERENCE_HARMONIZED_SNPS}"
    fi

    # Update variables for downstream steps
    BIOBANK_STANDARDIZED="${BIOBANK_HARMONIZED}"
    REFERENCE_STANDARDIZED="${REFERENCE_HARMONIZED}"
else
    print_header "Skipping WRayner Harmonization"
    print_status "Using plink2 SNP ID standardization instead..."

    # Standardize SNP IDs without WRayner
    BIOBANK_STANDARDIZED="${TEMP_DIR}/biobank_standardized"
    REFERENCE_STANDARDIZED="${TEMP_DIR}/reference_standardized"

    if [[ ! -f "${BIOBANK_STANDARDIZED}.bed" ]]; then
        print_status "Standardizing biobank SNP IDs..."
        ${PLINK2} --bfile ${BIOBANK_FILTERED} \
            --set-all-var-ids '@:#:$r:$a' \
            --new-id-max-allele-len 100 missing \
            --memory ${MEMORY} \
            --threads ${THREADS} \
            --make-bed \
            --out ${BIOBANK_STANDARDIZED}
        print_success "Biobank standardized"
    fi

    if [[ ! -f "${REFERENCE_STANDARDIZED}.bed" ]]; then
        print_status "Standardizing reference SNP IDs..."
        ${PLINK2} --bfile ${REFERENCE_FILTERED} \
            --set-all-var-ids '@:#:$r:$a' \
            --new-id-max-allele-len 100 missing \
            --memory ${MEMORY} \
            --threads ${THREADS} \
            --make-bed \
            --out ${REFERENCE_STANDARDIZED}
        print_success "Reference standardized"
    fi
fi

# ============================================================================
# Step 6: LD Pruning (optional)
# ============================================================================
if [[ "$SKIP_LD_PRUNE" != "true" ]]; then
    print_header "LD Pruning"

    echo "LD pruning reference SNPs and applying to biobank..."
    echo "  Parameters: ${LD_WINDOW}kb window, step ${LD_STEP}, r²=${LD_R2}"

    REFERENCE_PRUNE_PREFIX="${TEMP_DIR}/reference_prune"
    REFERENCE_PRUNED_PREFIX="${TEMP_DIR}/reference_pruned"
    BIOBANK_PRUNED_PREFIX="${TEMP_DIR}/biobank_pruned"

    if [[ ! -f "${REFERENCE_PRUNE_PREFIX}.prune.in" ]]; then
        print_status "Computing prune list on reference..."
        ${PLINK2} --bfile ${REFERENCE_STANDARDIZED} \
            --indep-pairwise ${LD_WINDOW} ${LD_STEP} ${LD_R2} \
            --memory ${MEMORY} \
            --threads ${THREADS} \
            --out ${REFERENCE_PRUNE_PREFIX}
    else
        print_success "Prune list already exists"
    fi

    PRUNE_IN_COUNT=$(wc -l < "${REFERENCE_PRUNE_PREFIX}.prune.in")
    print_success "SNPs retained after LD pruning: ${PRUNE_IN_COUNT}"

    if [[ ! -f "${REFERENCE_PRUNED_PREFIX}.bed" ]]; then
        print_status "Applying prune list to reference..."
        ${PLINK2} --bfile ${REFERENCE_STANDARDIZED} \
            --extract ${REFERENCE_PRUNE_PREFIX}.prune.in \
            --make-bed \
            --out ${REFERENCE_PRUNED_PREFIX}
    fi

    if [[ ! -f "${BIOBANK_PRUNED_PREFIX}.bed" ]]; then
        print_status "Applying prune list to biobank..."
        ${PLINK2} --bfile ${BIOBANK_STANDARDIZED} \
            --extract ${REFERENCE_PRUNE_PREFIX}.prune.in \
            --memory ${MEMORY} \
            --threads ${THREADS} \
            --make-bed \
            --out ${BIOBANK_PRUNED_PREFIX}
    fi

    # Update downstream variables
    REFERENCE_PLINK_PREFIX="${REFERENCE_PRUNED_PREFIX}"
    BIOBANK_PLINK_PREFIX="${BIOBANK_PRUNED_PREFIX}"
else
    print_header "Skipping LD Pruning"
    REFERENCE_PLINK_PREFIX="${REFERENCE_STANDARDIZED}"
    BIOBANK_PLINK_PREFIX="${BIOBANK_STANDARDIZED}"
fi

# ============================================================================
# Step 7: Find SNP Intersection and Check Allele Compatibility
# ============================================================================
print_header "SNP Intersection and Allele Compatibility"

if [[ -f "${TEMP_DIR}/final_common_snps.txt" ]] && [[ -f "${TEMP_DIR}/flip_list.txt" ]] && [[ -f "${TEMP_DIR}/exclude_list.txt" ]]; then
    print_success "SNP intersection already computed, skipping..."
    FINAL_SNP_COUNT=$(wc -l < "${TEMP_DIR}/final_common_snps.txt")
    FLIP_COUNT=$(wc -l < "${TEMP_DIR}/flip_list.txt")
    EXCLUDE_COUNT=$(wc -l < "${TEMP_DIR}/exclude_list.txt")
    echo "  Final SNPs: $FINAL_SNP_COUNT, Flip: $FLIP_COUNT, Exclude: $EXCLUDE_COUNT"
else
    print_status "Finding SNP intersection..."

    awk '{print $2}' "${REFERENCE_PLINK_PREFIX}.bim" | sort > "${TEMP_DIR}/reference_snps.txt"
    awk '{print $2}' "${BIOBANK_PLINK_PREFIX}.bim" | sort > "${TEMP_DIR}/biobank_snps.txt"

    REFERENCE_SNP_COUNT=$(wc -l < "${TEMP_DIR}/reference_snps.txt")
    BIOBANK_SNP_COUNT=$(wc -l < "${TEMP_DIR}/biobank_snps.txt")

    echo "  Extracted SNP lists:"
    echo "    Reference: $REFERENCE_SNP_COUNT SNPs"
    echo "    Biobank: $BIOBANK_SNP_COUNT SNPs"

    comm -12 "${TEMP_DIR}/reference_snps.txt" "${TEMP_DIR}/biobank_snps.txt" > "${TEMP_DIR}/common_snps.txt"

    COMMON_SNP_COUNT=$(wc -l < "${TEMP_DIR}/common_snps.txt")
    print_success "Found $COMMON_SNP_COUNT common SNPs"

    if [[ $COMMON_SNP_COUNT -lt 50000 ]]; then
        print_error "Less than 50K common SNPs! Data may be incompatible."
        exit 1
    elif [[ $COMMON_SNP_COUNT -lt 100000 ]]; then
        print_warning "Less than 100K common SNPs! Check data compatibility."
    fi

    print_status "Checking allele consistency..."

    python3 << EOF
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path("${PROJECT_ROOT}/src")))

from manifold_genetics.utils.io import check_allele_compatibility

# Read inputs
reference_bim = Path("${REFERENCE_PLINK_PREFIX}.bim")
biobank_bim = Path("${BIOBANK_PLINK_PREFIX}.bim")
common_snps_file = Path("${TEMP_DIR}/common_snps.txt")
output_dir = Path("${TEMP_DIR}")

# Read common SNPs
with open(common_snps_file, 'r') as f:
    common_snps = {line.strip() for line in f if line.strip()}

# Check allele compatibility
exact_matches, need_flip, incompatible = check_allele_compatibility(
    reference_bim,
    biobank_bim,
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
    echo "    SNPs to flip: $FLIP_COUNT"
    echo "    SNPs to exclude: $EXCLUDE_COUNT"
    echo "    Usable SNPs: $USABLE_SNPS"

    # Create final common SNPs list (excluding incompatible)
    if [[ $EXCLUDE_COUNT -gt 0 ]]; then
        print_status "Removing excluded SNPs from common list..."
        comm -23 <(sort "${TEMP_DIR}/common_snps.txt") <(sort "${TEMP_DIR}/exclude_list.txt") > "${TEMP_DIR}/final_common_snps.txt"
    else
        cp "${TEMP_DIR}/common_snps.txt" "${TEMP_DIR}/final_common_snps.txt"
    fi

    FINAL_SNP_COUNT=$(wc -l < "${TEMP_DIR}/final_common_snps.txt")
    print_success "Final SNP count: $FINAL_SNP_COUNT"
fi

# ============================================================================
# Step 8: Create Final Intersected PLINK Files
# ============================================================================
print_header "Creating Final Datasets"

print_status "Creating intersected PLINK files..."

# Create reference intersected dataset (fit_subset)
if [[ ! -f "${TEMP_DIR}/reference_intersected.bed" ]]; then
    print_status "Creating reference intersected dataset..."
    REFERENCE_CMD="${PLINK2} --bfile ${REFERENCE_PLINK_PREFIX} --extract ${TEMP_DIR}/final_common_snps.txt"
    if [[ $FLIP_COUNT -gt 0 ]]; then
        REFERENCE_CMD="$REFERENCE_CMD --flip ${TEMP_DIR}/flip_list.txt"
    fi
    REFERENCE_CMD="$REFERENCE_CMD --make-bed --out ${TEMP_DIR}/reference_intersected"
    eval $REFERENCE_CMD
else
    print_success "Reference intersected dataset already exists"
fi

# Create biobank intersected dataset (project_subset)
if [[ ! -f "${TEMP_DIR}/biobank_intersected.bed" ]]; then
    print_status "Creating biobank intersected dataset..."
    ${PLINK2} --bfile ${BIOBANK_PLINK_PREFIX} \
        --extract ${TEMP_DIR}/final_common_snps.txt \
        --memory ${MEMORY} \
        --threads ${THREADS} \
        --make-bed \
        --out ${TEMP_DIR}/biobank_intersected
else
    print_success "Biobank intersected dataset already exists"
fi

print_success "Intersected datasets created"

# Create final output files
print_status "Creating final processed subsets..."

# Create fit_subset (reference)
if [[ ! -f "${OUTPUT_DIR}/fit_subset.bed" ]]; then
    cp "${TEMP_DIR}/reference_intersected.bed" "${OUTPUT_DIR}/fit_subset.bed"
    cp "${TEMP_DIR}/reference_intersected.bim" "${OUTPUT_DIR}/fit_subset.bim"
    cp "${TEMP_DIR}/reference_intersected.fam" "${OUTPUT_DIR}/fit_subset.fam"
else
    print_warning "fit_subset already exists, skipping"
fi

# Create project_subset (biobank)
if [[ ! -f "${OUTPUT_DIR}/project_subset.bed" ]]; then
    cp "${TEMP_DIR}/biobank_intersected.bed" "${OUTPUT_DIR}/project_subset.bed"
    cp "${TEMP_DIR}/biobank_intersected.bim" "${OUTPUT_DIR}/project_subset.bim"
    cp "${TEMP_DIR}/biobank_intersected.fam" "${OUTPUT_DIR}/project_subset.fam"
else
    print_warning "project_subset already exists, skipping"
fi

# Get final counts
FIT_SAMPLES=$(get_sample_count "${OUTPUT_DIR}/fit_subset")
PROJECT_SAMPLES=$(get_sample_count "${OUTPUT_DIR}/project_subset")
FINAL_SNP_COUNT=$(get_snp_count "${OUTPUT_DIR}/fit_subset")

# ============================================================================
# Summary
# ============================================================================
print_header "Preprocessing Complete!"

echo "Generated files in ${OUTPUT_DIR}:"
echo "  fit_subset.{bed,bim,fam}     (Reference: $FIT_SAMPLES samples, $FINAL_SNP_COUNT SNPs)"
echo "  project_subset.{bed,bim,fam} (Biobank: $PROJECT_SAMPLES samples, $FINAL_SNP_COUNT SNPs)"
echo ""
echo "Steps completed:"
echo "  [✓] Filtering (MAF>${MAF_THRESHOLD}, geno<${GENO_THRESHOLD}, no indels)"
if [[ "$SKIP_DEDUP" != "true" ]]; then
    echo "  [✓] Deduplication (remove multi-allelic positions)"
fi
if [[ "$SKIP_GIAB" != "true" ]]; then
    echo "  [✓] GIAB difficult regions excluded"
fi
if [[ "$SKIP_HLA" != "true" ]]; then
    echo "  [✓] HLA/MHC region excluded"
fi
if [[ "$SKIP_WRAYNER" != "true" ]]; then
    echo "  [✓] WRayner harmonization (TOPMed reference)"
else
    echo "  [✓] SNP ID standardization (chr:pos:ref:alt)"
fi
if [[ "$SKIP_LD_PRUNE" != "true" ]]; then
    echo "  [✓] LD pruning (${LD_WINDOW}kb, r²=${LD_R2})"
fi
echo "  [✓] SNP intersection and allele compatibility check"
echo "  [✓] Final intersected datasets created"
echo ""
echo "Intermediate files in ${TEMP_DIR}/ can be deleted to save space."
echo ""
