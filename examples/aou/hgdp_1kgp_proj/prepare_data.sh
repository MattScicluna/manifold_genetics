#!/bin/bash
#
# Prepare AoU-HGDP cross-projection data
#
# This script:
# 1. Downloads/prepares HGDP+1KGP reference data (Steps 1-4)
# 2. Downloads AoU genotype data via shared script (Step 5)
# 3. Calls shared preprocessing for filtering, harmonization, intersection (Steps 6-13)
# 4. Creates labels and colormaps (Step 14)
#

set -e

# AoU specific
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-32}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SHARED_PREPROCESS="${PROJECT_ROOT}/examples/_shared/preprocessing/preprocess_cross_projection.sh"

# Source common utilities
source "${PROJECT_ROOT}/examples/_shared/preprocessing/common.sh"

# Configuration
CPU_CORES="${SLURM_CPUS_PER_TASK:-4}"
REF_DATA_URL="gs://fc-secure-47ccf5a8-b9ba-460a-aa03-dea8d260953b/Data/1KGPHGDP.tar.gz"
GOOGLE_PROJECT="${GOOGLE_PROJECT:-}"
CDR_VERSION="${WORKSPACE_CDR:-}"

# =============================================================================
# Filtering parameters (passed to shared preprocessing script)
# =============================================================================
MAF_THRESHOLD="${MAF_THRESHOLD:-0.01}"
GENO_THRESHOLD="${GENO_THRESHOLD:-0.05}"
LD_WINDOW="${LD_WINDOW:-150}"
LD_STEP="${LD_STEP:-1}"
LD_R2="${LD_R2:-0.05}"

# Directories
TEMP_DIR="${DATA_DIR}/temp"
REF_DIR="${TEMP_DIR}/1KGPHGDP"
SHARED_AOU_DIR="${SCRIPT_DIR}/../shared/data/AllofUs_V8"
SHARED_META_DIR="${SCRIPT_DIR}/../shared/data/Metadata"

print_header "AoU-HGDP Data Preparation"

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
# Find plink (needed for Step 4: split by chromosome)
# =============================================================================
print_header "Checking Required Tools"

print_status "Looking for plink (v1.9)..."
if ! find_plink "$PROJECT_ROOT"; then
    exit 1
fi

# =============================================================================
# STEPS 1-4: Reference Data Processing (HGDP+1KGP)
# =============================================================================
print_header "Steps 1-4: HGDP+1KGP Reference Data"

TAR_PATH="${TEMP_DIR}/1KGPHGDP.tar.gz"

# 1. Download
if [[ ! -f "${TAR_PATH}" ]]; then
    print_status "Downloading HGDP+1KGP reference data..."
    echo "  Source: ${REF_DATA_URL}"
    gsutil cp "${REF_DATA_URL}" "${TEMP_DIR}/"
else
    print_success "Reference data archive already exists"
fi

# 2. Extract
if [[ ! -f "${REF_DIR}/extractedChrAllUnpruned.bed" ]]; then
    print_status "Extracting reference data..."
    tar -xf "${TAR_PATH}" --directory "${TEMP_DIR}/"
else
    print_success "Reference data already extracted"
fi

# 3. Fix BIM file (add 'chr' prefix)
BIM_PATH="${REF_DIR}/extractedChrAllUnpruned.bim"
ORIG_BIM_PATH="${REF_DIR}/extractedChrAllUnpruned.Original.bim"

if [[ ! -f "${ORIG_BIM_PATH}" ]]; then
    print_status "Fixing BIM file chromosome prefixes..."
    mv "${BIM_PATH}" "${ORIG_BIM_PATH}"
    awk '{print "chr"$1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6}' "${ORIG_BIM_PATH}" > "${BIM_PATH}"
else
    print_success "BIM file already fixed"
fi

# 4. Split by chromosome (1-22) - optional, for parallel processing
print_status "Splitting reference data by chromosome (using ${CPU_CORES} cores)..."
if [[ ! -f "${REF_DIR}/extractedChr22.bed" ]]; then
    export PLINK
    seq 1 22 | xargs -I {} -P "${CPU_CORES}" sh -c \
        "${PLINK} --bfile ${REF_DIR}/extractedChrAllUnpruned \
               --keep-allele-order --allow-no-sex --chr {} \
               --make-bed --out ${REF_DIR}/extractedChr{}"
else
    print_success "Reference data already split by chromosome"
fi

print_success "HGDP+1KGP reference data ready"

# =============================================================================
# STEP 5: All of Us Data Download (shared)
# =============================================================================
print_header "Step 5: All of Us Data Download"

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
    print_success "Shared AoU data already downloaded at: ${SHARED_AOU_DIR}"
    echo "  Skipping download (shared across all AoU experiments)"
else
    print_status "Downloading AoU data to shared location..."
    echo "  This will be cached for all AoU experiments"
    bash "${SCRIPT_DIR}/../shared/download_aou_data.sh"
fi

# =============================================================================
# STEPS 6-13: Shared Preprocessing
# =============================================================================
print_header "Steps 6-13: Shared Preprocessing"

# Check if preprocessing already complete
if [[ -f "${DATA_DIR}/fit_subset.bed" ]] && [[ -f "${DATA_DIR}/project_subset.bed" ]]; then
    print_success "Preprocessing already complete"
    echo "  fit_subset and project_subset exist in ${DATA_DIR}/"
    echo "  Delete them to re-run preprocessing"
else
    print_status "Running shared preprocessing script..."
    echo "  Reference: ${REF_DIR}/extractedChrAllUnpruned"
    echo "  Biobank: ${SHARED_AOU_DIR}/extractedChrAll"
    echo ""

    bash "${SHARED_PREPROCESS}" \
        --reference-plink "${REF_DIR}/extractedChrAllUnpruned" \
        --biobank-plink "${SHARED_AOU_DIR}/extractedChrAll" \
        --output-dir "${DATA_DIR}" \
        --temp-dir "${TEMP_DIR}" \
        --threads "${CPU_CORES}" \
        --maf "${MAF_THRESHOLD}" \
        --geno "${GENO_THRESHOLD}" \
        --ld-window "${LD_WINDOW}" \
        --ld-step "${LD_STEP}" \
        --ld-r2 "${LD_R2}" \
        --reference-has-chr-prefix

    print_success "Shared preprocessing complete"
fi

# =============================================================================
# STEP 14: Create Labels and Colormaps
# =============================================================================
print_header "Step 14: Labels and Colormaps"

# Create HGDP labels
print_status "Creating HGDP labels..."

if [[ -f "${DATA_DIR}/hgdp_labels.csv" ]]; then
    print_success "HGDP labels already exist"
    HGDP_LABEL_COUNT=$(wc -l < "${DATA_DIR}/hgdp_labels.csv")
    echo "  Existing labels: $((HGDP_LABEL_COUNT - 1)) samples"
else
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
print(f"  Saved HGDP labels: {len(hgdp_labels)} samples")
print(f"  Unique populations: {hgdp_labels['Population'].nunique()}")
EOF
fi

# Create AoU labels
print_status "Creating AoU labels..."

if [[ -f "${DATA_DIR}/aou_labels.csv" ]]; then
    print_success "AoU labels already exist"
    AOU_LABEL_COUNT=$(wc -l < "${DATA_DIR}/aou_labels.csv")
    echo "  Existing labels: $((AOU_LABEL_COUNT - 1)) samples"
else
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

# Select relevant columns
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
print(f"  Saved AoU labels: {len(aou_labels)} samples")
print(f"  Columns: {', '.join(columns_to_keep)}")

if len(aou_labels) < len(available_samples):
    missing = len(available_samples) - len(aou_labels)
    print(f"  Warning: {missing} samples in intersected data have no metadata")
EOF
    else
        print_error "AoU metadata not found at: $AOU_METADATA"
        echo ""
        echo "  This file is created by the AoU data download script."
        echo "  Ensure WORKSPACE_CDR is set and re-run:"
        echo ""
        echo "    bash ${PROJECT_ROOT}/examples/aou/shared/download_aou_data.sh"
        echo ""
        exit 1
    fi
fi

print_success "Label files created"

# =============================================================================
# Summary
# =============================================================================
print_header "Data Preparation Complete!"

# Get final counts
FIT_SAMPLES=$(get_sample_count "${DATA_DIR}/fit_subset")
PROJECT_SAMPLES=$(get_sample_count "${DATA_DIR}/project_subset")
FINAL_SNPS=$(get_snp_count "${DATA_DIR}/fit_subset")

echo "Generated files in ${DATA_DIR}:"
echo "  fit_subset.{bed,bim,fam}     (HGDP+1KGP: $FIT_SAMPLES samples, $FINAL_SNPS SNPs)"
echo "  project_subset.{bed,bim,fam} (AoU: $PROJECT_SAMPLES samples, $FINAL_SNPS SNPs)"
echo "  hgdp_labels.csv              (HGDP sample metadata)"
echo "  aou_labels.csv               (AoU sample metadata)"
echo ""
echo "Filtering parameters used:"
echo "  MAF threshold: ${MAF_THRESHOLD}"
echo "  Geno threshold: ${GENO_THRESHOLD}"
echo "  LD pruning: ${LD_WINDOW}kb window, step ${LD_STEP}, r²=${LD_R2}"
echo ""
echo "To adjust parameters, modify environment variables and re-run:"
echo "  MAF_THRESHOLD=0.01 GENO_THRESHOLD=0.01 LD_R2=0.2 bash prepare_data.sh"
echo ""
echo "Next step: Run the cross-projection pipeline"
echo "  bash run_pipeline.sh"
echo ""
