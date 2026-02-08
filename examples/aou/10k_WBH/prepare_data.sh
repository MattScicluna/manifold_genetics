#!/bin/bash
#
# AoU 10K White + 10K Black + 10K Hispanic Subset Data Preparation
#
# This wrapper script:
# 1. Reads sample IDs from intersected AoU data (from hgdp_1kgp_proj)
# 2. Creates fit_samples.txt with 10K White + 10K Black + 10K Hispanic samples
# 3. Calls generic subset script to create PLINK subsets
# 4. Creates fit_labels.csv and project_labels.csv
#
# Strategy:
# - Fit Subset: 10K White + 10K Black/African American + 10K Hispanic/Latino
# - Project Subset: The entire intersected dataset
#
# Usage:
#   bash examples/aou/10k_WBH/prepare_data.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TEMP_DIR="${DATA_DIR}/temp"

# Source common utilities
source "${PROJECT_ROOT}/examples/_shared/preprocessing/common.sh"

# Configuration
CPU_CORES="${SLURM_CPUS_PER_TASK:-4}"
GOOGLE_PROJECT="${GOOGLE_PROJECT:-}"
CDR_VERSION="${WORKSPACE_CDR:-}"

# Shared AoU data directories
SHARED_META_DIR="${SCRIPT_DIR}/../shared/data/Metadata"

# Use intersected AoU data from hgdp_1kgp_proj
INTERSECTED_AOU="${SCRIPT_DIR}/../hgdp_1kgp_proj/data/project_subset"

print_header "AoU 10K White + 10K Black + 10K Hispanic Data Preparation"

echo "Configuration:"
echo "  CPU cores: ${CPU_CORES}"
echo "  Google project: ${GOOGLE_PROJECT:-'Not set'}"
echo "  CDR version: ${CDR_VERSION:-'Not set'}"
echo ""

# Create directories
mkdir -p "${DATA_DIR}" "${TEMP_DIR}"

# =============================================================================
# STEP 1: Check for Intersected AoU Data
# =============================================================================
print_subheader "Step 1: Check for Intersected AoU Data"

if [[ ! -f "${INTERSECTED_AOU}.bed" ]]; then
    print_error "Intersected AoU data not found!"
    echo "  Expected: ${INTERSECTED_AOU}.bed"
    echo "  Please run the hgdp_1kgp_proj pipeline first:"
    echo "    bash examples/aou/hgdp_1kgp_proj/prepare_data.sh"
    exit 1
fi

AOU_SAMPLES=$(get_sample_count "${INTERSECTED_AOU}")
COMMON_SNPS=$(get_snp_count "${INTERSECTED_AOU}")

print_success "Found intersected AoU data:"
echo "    Samples: $AOU_SAMPLES"
echo "    SNPs: $COMMON_SNPS"
echo ""

# =============================================================================
# STEP 2: Define Fit Subset (Shared Selection Script)
# =============================================================================
print_subheader "Step 2: Create Fit Sample List"

AOU_METADATA="${SHARED_META_DIR}/DemographicData.tsv"

if [[ ! -f "$AOU_METADATA" ]]; then
    print_error "AoU metadata not found: $AOU_METADATA"
    echo "  Please run the hgdp_1kgp_proj data download first."
    exit 1
fi

FIT_SAMPLES_FILE="${DATA_DIR}/fit_samples.txt"

if [[ -f "$FIT_SAMPLES_FILE" ]]; then
    print_success "Fit samples list already exists"
    FIT_SAMPLES_COUNT=$(wc -l < "$FIT_SAMPLES_FILE")
    echo "    Existing samples: $FIT_SAMPLES_COUNT"
else
    echo "  Selecting samples for fit subset..."
    echo "  Target: 10,000 White + 10,000 Black/African American + 10,000 Hispanic/Latino"

    python3 "${SCRIPT_DIR}/../shared/select_samples.py" \
        --metadata "$AOU_METADATA" \
        --fam "${INTERSECTED_AOU}.fam" \
        --output "$FIT_SAMPLES_FILE" \
        --group "White|European:10000" \
        --group "Black or African American:10000" \
        --group "Hispanic or Latino:10000" \
        --seed 42
fi

# =============================================================================
# STEP 3: Call Generic Subset Script
# =============================================================================
print_subheader "Step 3: Create PLINK Subsets"

# Note: project_samples not provided means use all samples from PLINK (entire dataset)
bash "${PROJECT_ROOT}/examples/generic/subset/prepare_data.sh" \
    --plink "$INTERSECTED_AOU" \
    --fit-samples "$FIT_SAMPLES_FILE" \
    --output-dir "$DATA_DIR" \
    --memory 100000 \
    --threads "$CPU_CORES"

# =============================================================================
# STEP 4: Create Labels
# =============================================================================
print_subheader "Step 4: Create Label Files"

# Create labels for both subsets
for subset in "fit" "project"; do
    LABEL_FILE="${DATA_DIR}/${subset}_labels.csv"
    if [[ -f "$LABEL_FILE" ]]; then
        print_success "${subset}_labels.csv already exists"
        continue
    fi

    echo "  Creating ${subset} subset labels..."
    python3 << EOF
import pandas as pd

metadata = pd.read_csv("${AOU_METADATA}", sep='\t', low_memory=False)
metadata['sample_id'] = metadata['person_id'].astype(str)

fam = pd.read_csv("${DATA_DIR}/${subset}_subset.fam", sep=r'\s+', header=None,
                  names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Phenotype'])
subset_samples = set(fam['IID'].astype(str))

labels = metadata[metadata['sample_id'].isin(subset_samples)].copy()
columns = ['sample_id']
for col in ['race_ethnicity', 'sex_at_birth', 'age']:
    if col in labels.columns:
        columns.append(col)

labels[columns].to_csv("${LABEL_FILE}", index=False)
print(f"    Created {len(labels)} labels with columns: {', '.join(columns)}")
EOF
done

print_success "Labels created"

# =============================================================================
# Summary
# =============================================================================
FIT_SAMPLES=$(get_sample_count "${DATA_DIR}/fit_subset")
PROJECT_SAMPLES=$(get_sample_count "${DATA_DIR}/project_subset")
FINAL_SNPS=$(get_snp_count "${DATA_DIR}/fit_subset")

print_header "Data Preparation Complete!"

echo "Generated files in ${DATA_DIR}:"
echo "  fit_subset.{bed,bim,fam}     (10K W + 10K B + 10K H: $FIT_SAMPLES samples, $FINAL_SNPS SNPs)"
echo "  project_subset.{bed,bim,fam} (Full dataset: $PROJECT_SAMPLES samples, $FINAL_SNPS SNPs)"
echo "  fit_labels.csv               (Fit sample metadata)"
echo "  project_labels.csv           (Project sample metadata)"
echo ""
echo "Next step: Run the subsample pipeline"
echo "  bash run_pipeline.sh"
echo ""
