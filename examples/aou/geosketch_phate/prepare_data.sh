#!/bin/bash
#
# AoU Geosketch PHATE - Data Preparation
#
# Selects a geometrically representative fit subset from the intersected AoU
# data using geosketch on PCA coordinates, then creates matching PLINK subsets
# and label files.
#
# Prerequisites:
#   1. Run examples/aou/hgdp_1kgp_proj/prepare_data.sh  (intersected PLINK data)
#   2. Run examples/aou/10k_WBH/run_pipeline.sh          (produces PCA for AoU samples)
#
# The PCA CSV used for geosketch is from the 10k_WBH pipeline
# (PCA fit on 10K WBH, projected to all AoU samples):
#   examples/aou/10k_WBH/outputs/pca/transform_pca_20.csv
#
# The same PCA coordinates are written to outputs/pca/fit_pca_20.csv (for the
# selected samples) so that run_pipeline.sh --skip-pca uses the correct space.
#
# Usage:
#   bash examples/aou/geosketch_phate/prepare_data.sh
#
# Override defaults via environment variables:
#   N_SKETCH=50000 bash examples/aou/geosketch_phate/prepare_data.sh
#   PCA_CSV=/path/to/pca.csv bash examples/aou/geosketch_phate/prepare_data.sh
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

# Source intersected AoU data (from hgdp_1kgp_proj) — used for PLINK subsets + admixture
INTERSECTED_AOU="${SCRIPT_DIR}/../hgdp_1kgp_proj/data/project_subset"

# PCA CSV for geosketch: 10k_WBH projected PCA coordinates for all AoU samples
# This matches the approach used for UKBB (same PCA space for geosketch and PHATE)
PCA_CSV="${PCA_CSV:-${SCRIPT_DIR}/../10k_WBH/outputs/pca/transform_pca_20.csv}"

# Geosketch parameters
N_SKETCH="${N_SKETCH:-60000}"
N_PCS="${N_PCS:-20}"
SEED="${SEED:-42}"

print_header "AoU Geosketch PHATE Data Preparation"

echo "Configuration:"
echo "  CPU cores:       ${CPU_CORES}"
echo "  Geosketch N:     ${N_SKETCH}"
echo "  PCs for sketch:  ${N_PCS}"
echo "  Seed:            ${SEED}"
echo "  PCA CSV:         ${PCA_CSV}"
echo ""

# Create directories
mkdir -p "${DATA_DIR}" "${TEMP_DIR}"

# =============================================================================
# STEP 1: Check Prerequisites
# =============================================================================
print_subheader "Step 1: Check Prerequisites"

if [[ ! -f "${INTERSECTED_AOU}.bed" ]]; then
    print_error "Intersected AoU PLINK data not found!"
    echo "  Expected: ${INTERSECTED_AOU}.bed"
    echo ""
    echo "  Please run the cross-projection pipeline first:"
    echo "    bash examples/aou/hgdp_1kgp_proj/prepare_data.sh"
    exit 1
fi

if [[ ! -f "${PCA_CSV}" ]]; then
    print_error "PCA CSV not found for geosketch!"
    echo "  Expected: ${PCA_CSV}"
    echo ""
    echo "  This file is produced by the 10k_WBH pipeline."
    echo "  Please run it first (or submit as a batch job):"
    echo ""
    echo "    bash examples/_shared/submit_batch.sh \\"
    echo "        examples/aou/10k_WBH/run_pipeline.sh"
    echo ""
    echo "  Alternatively, provide a different PCA CSV via:"
    echo "    PCA_CSV=/path/to/pca.csv bash prepare_data.sh"
    exit 1
fi

AOU_SAMPLES=$(get_sample_count "${INTERSECTED_AOU}")
COMMON_SNPS=$(get_snp_count "${INTERSECTED_AOU}")

print_success "Found intersected AoU data:"
echo "    Samples: $AOU_SAMPLES"
echo "    SNPs:    $COMMON_SNPS"
print_success "Found PCA CSV: ${PCA_CSV}"
echo ""

# =============================================================================
# STEP 2: Geosketch Sample Selection
# =============================================================================
print_subheader "Step 2: Geosketch Sample Selection"

FIT_SAMPLES_FILE="${DATA_DIR}/fit_samples.txt"

if [[ -f "$FIT_SAMPLES_FILE" ]]; then
    print_success "Fit samples list already exists"
    FIT_SAMPLES_COUNT=$(wc -l < "$FIT_SAMPLES_FILE")
    echo "    Existing samples: $FIT_SAMPLES_COUNT"
else
    echo "  Selecting ${N_SKETCH} geometrically representative samples..."

    python3 "${PROJECT_ROOT}/examples/_shared/select_samples_geosketch.py" \
        --pca "${PCA_CSV}" \
        --fam "${INTERSECTED_AOU}.fam" \
        --output "${FIT_SAMPLES_FILE}" \
        --n-samples "${N_SKETCH}" \
        --n-pcs "${N_PCS}" \
        --seed "${SEED}"
fi

# =============================================================================
# STEP 3: Create PLINK Subsets
# =============================================================================
print_subheader "Step 3: Create PLINK Subsets"

# Note: project_samples not provided → use all samples (entire intersected dataset)
bash "${PROJECT_ROOT}/examples/generic/subset/prepare_data.sh" \
    --plink "$INTERSECTED_AOU" \
    --fit-samples "$FIT_SAMPLES_FILE" \
    --output-dir "$DATA_DIR" \
    --memory 100000 \
    --threads "$CPU_CORES"

# =============================================================================
# STEP 4: Create Label Files
# =============================================================================
print_subheader "Step 4: Create Label Files"

AOU_METADATA="${SHARED_META_DIR}/DemographicData.tsv"

if [[ ! -f "$AOU_METADATA" ]]; then
    print_error "AoU metadata not found: $AOU_METADATA"
    echo ""
    echo "  This file is created by the AoU data download script."
    echo "  Ensure WORKSPACE_CDR is set and re-run:"
    echo ""
    echo "    bash ${PROJECT_ROOT}/examples/aou/shared/download_aou_data.sh"
    echo ""
    exit 1
fi

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
# STEP 5: Create fit PCA file for --skip-pca
# =============================================================================
print_subheader "Step 5: Create fit PCA file"

# Subset PCA_CSV to the selected fit samples so that run_pipeline.sh --skip-pca
# uses the same 10k_WBH coordinate space that geosketch ran in.
OUTPUT_PCA_DIR="${SCRIPT_DIR}/outputs/pca"
FIT_PCA_FILE="${OUTPUT_PCA_DIR}/fit_pca_${N_PCS}.csv"

mkdir -p "${OUTPUT_PCA_DIR}"

if [[ -f "$FIT_PCA_FILE" ]]; then
    print_success "fit_pca_${N_PCS}.csv already exists ($(wc -l < "$FIT_PCA_FILE") lines)"
else
    echo "  Subsetting PCA coordinates to fit samples..."
    python3 << EOF
import pandas as pd

fit_samples_file = "${FIT_SAMPLES_FILE}"
pca_csv = "${PCA_CSV}"
output_file = "${FIT_PCA_FILE}"
n_pcs = ${N_PCS}

# Read selected IIDs from fit_samples.txt (FID TAB IID)
selected_iids = set()
with open(fit_samples_file) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            selected_iids.add(str(parts[1]))

# Load and subset the PCA CSV
pca_df = pd.read_csv(pca_csv)
pca_df['sample_id'] = pca_df['sample_id'].astype(str)

dim_cols = [c for c in pca_df.columns if c != 'sample_id'][:n_pcs]
fit_pca = pca_df[pca_df['sample_id'].isin(selected_iids)][['sample_id'] + dim_cols].copy()
fit_pca.to_csv(output_file, index=False)
print(f"    Written {len(fit_pca)} samples x {len(dim_cols)} PCs to {output_file}")
EOF
    print_success "fit_pca_${N_PCS}.csv created"
fi

# =============================================================================
# Summary
# =============================================================================
FIT_SAMPLES=$(get_sample_count "${DATA_DIR}/fit_subset")
PROJECT_SAMPLES=$(get_sample_count "${DATA_DIR}/project_subset")
FINAL_SNPS=$(get_snp_count "${DATA_DIR}/fit_subset")

print_header "Data Preparation Complete!"

echo "Generated files in ${DATA_DIR}:"
echo "  fit_subset.{bed,bim,fam}     (geosketch N=${N_SKETCH}: $FIT_SAMPLES samples, $FINAL_SNPS SNPs)"
echo "  project_subset.{bed,bim,fam} (Full AoU dataset: $PROJECT_SAMPLES samples, $FINAL_SNPS SNPs)"
echo "  fit_labels.csv               (Fit sample metadata)"
echo "  project_labels.csv           (Project sample metadata)"
echo ""
echo "Next step: Run the PHATE pipeline"
echo "  bash run_pipeline.sh"
echo "  # or submit as a batch job:"
echo "  bash examples/_shared/submit_batch.sh examples/aou/geosketch_phate/run_pipeline.sh"
echo ""
