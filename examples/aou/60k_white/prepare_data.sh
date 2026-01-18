#!/bin/bash
#
# Prepare AoU 60K White + Non-White Subset Data
#
# Strategy:
# 1. Fit Subset: Random 60K White + ALL Non-White samples.
# 2. Project Subset: The entire dataset (intersected with HGDP).
#
# Optimization:
# - Performs sample selection in Python to avoid creating intermediate PLINK files.
# - Directly extracts the final fit subset from the source.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TEMP_DIR="${DATA_DIR}/temp"

# Configuration
CPU_CORES="${SLURM_CPUS_PER_TASK:-4}"
GOOGLE_PROJECT="${GOOGLE_PROJECT:-}"
CDR_VERSION="${WORKSPACE_CDR:-}"

# Shared AoU data directories
SHARED_AOU_DIR="${SCRIPT_DIR}/../shared/data/AllofUs_V8"
SHARED_META_DIR="${SCRIPT_DIR}/../shared/data/Metadata"

# Use intersected AoU data from hgdp_1kgp_proj
INTERSECTED_AOU="${SCRIPT_DIR}/../hgdp_1kgp_proj/data/project_subset"

echo "=========================================="
echo "  AoU 60K White + Non-White Data Preparation"
echo "=========================================="
echo ""
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
echo "=========================================="
echo "Step 1: Check for Intersected AoU Data"
echo "=========================================="
echo ""

if [[ ! -f "${INTERSECTED_AOU}.bed" ]]; then
    echo "  ✗ Intersected AoU data not found!"
    echo "  Please run the hgdp_1kgp_proj pipeline first."
    exit 1
fi

AOU_SAMPLES=$(wc -l < "${INTERSECTED_AOU}.fam")
COMMON_SNPS=$(wc -l < "${INTERSECTED_AOU}.bim")

echo "  ✓ Found intersected AoU data:"
echo "    Samples: $AOU_SAMPLES"
echo "    SNPs: $COMMON_SNPS"
echo ""

# =============================================================================
# STEP 2: Define Fit Subset (Python Selection)
# =============================================================================
echo "=========================================="
echo "Step 2: Define Fit Subset List"
echo "=========================================="

AOU_METADATA="${SHARED_META_DIR}/DemographicData.tsv"

if [[ ! -f "$AOU_METADATA" ]]; then
    echo "  ✗ AoU metadata not found!"
    exit 1
fi

echo "  Selecting samples for fit subset..."
echo "  Target: 60,000 White/European + All Other ancestries"

python3 << EOF
import pandas as pd
import numpy as np

# Configuration
TARGET_WHITE_COUNT = 60000
SEED = 42

# Read metadata
metadata = pd.read_csv("${AOU_METADATA}", sep='\t', low_memory=False)
metadata['sample_id'] = metadata['person_id'].astype(str)

# Read intersected AoU .fam to get available samples and FIDs
fam = pd.read_csv("${INTERSECTED_AOU}.fam", sep=r'\s+', header=None,
                  names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Phenotype'])
iid_to_fid = dict(zip(fam['IID'].astype(str), fam['FID'].astype(str)))
available_samples = set(fam['IID'].astype(str))

# Filter metadata for available samples
metadata = metadata[metadata['sample_id'].isin(available_samples)]

# Identify White vs Other
if 'race_ethnicity' in metadata.columns:
    white_mask = metadata['race_ethnicity'].str.contains('White', case=False, na=False) | \
                 metadata['race_ethnicity'].str.contains('European', case=False, na=False)
elif 'race' in metadata.columns:
    white_mask = metadata['race'].str.contains('White', case=False, na=False) | \
                 metadata['race'].str.contains('European', case=False, na=False)
else:
    print("  ✗ Could not find race/ethnicity column in metadata!")
    exit(1)

white_samples = metadata[white_mask]
other_samples = metadata[~white_mask]

print(f"    Total White/European samples available: {len(white_samples)}")
print(f"    Total Other samples available: {len(other_samples)}")

# Subsample White group
if len(white_samples) > TARGET_WHITE_COUNT:
    print(f"    Subsampling White group to {TARGET_WHITE_COUNT}...")
    white_fit = white_samples.sample(n=TARGET_WHITE_COUNT, random_state=SEED)
else:
    print(f"    Using all available White samples ({len(white_samples)})...")
    white_fit = white_samples

# Combine for Fit Subset
fit_samples = pd.concat([white_fit, other_samples])
print(f"    Final Fit Subset size: {len(fit_samples)} (White: {len(white_fit)}, Other: {len(other_samples)})")

# Write ID list for PLINK (FID IID)
with open("${DATA_DIR}/fit_samples.txt", 'w') as f:
    for sample_id in fit_samples['sample_id']:
        if sample_id in iid_to_fid:
            f.write(f"{iid_to_fid[sample_id]}\t{sample_id}\n")

print(f"  ✓ Saved fit sample list to ${DATA_DIR}/fit_samples.txt")
EOF

# =============================================================================
# STEP 3: Create Fit Subset (PLINK Extraction)
# =============================================================================
echo "=========================================="
echo "Step 3: Create Fit Subset (PLINK)"
echo "=========================================="

# Find plink2
PLINK2="${PROJECT_ROOT}/bin/plink2"
if [[ ! -f "$PLINK2" ]]; then PLINK2="plink2"; fi

if [[ ! -f "${DATA_DIR}/fit_subset.bed" ]]; then
    echo "  Extracting fit subset from intersected data..."
    ${PLINK2} --bfile ${INTERSECTED_AOU} \
        --keep ${DATA_DIR}/fit_samples.txt \
        --memory 100000 \
        --make-bed \
        --out ${DATA_DIR}/fit_subset
else
    echo "  Fit subset already exists"
fi

FIT_SAMPLES=$(wc -l < "${DATA_DIR}/fit_subset.fam")
echo "  ✓ Fit subset created: $FIT_SAMPLES samples"
echo ""

# =============================================================================
# STEP 4: Create Project Subset (Entire Dataset)
# =============================================================================
echo "=========================================="
echo "Step 4: Create Project Subset (Entire Dataset)"
echo "=========================================="

# For the project step, we project the ENTIRE dataset (including those used in fit).
# We simply copy the intersected dataset to the expected project_subset path.

if [[ ! -f "${DATA_DIR}/project_subset.bed" ]]; then
    echo "  Copying intersected data to project subset..."
    cp "${INTERSECTED_AOU}.bed" "${DATA_DIR}/project_subset.bed"
    cp "${INTERSECTED_AOU}.bim" "${DATA_DIR}/project_subset.bim"
    cp "${INTERSECTED_AOU}.fam" "${DATA_DIR}/project_subset.fam"
else
    echo "  Project subset already exists"
fi

PROJECT_SAMPLES=$(wc -l < "${DATA_DIR}/project_subset.fam")
echo "  ✓ Project subset created: $PROJECT_SAMPLES samples"
echo ""

# =============================================================================
# STEP 5: Create Labels
# =============================================================================
echo "=========================================="
echo "Step 5: Create Labels"
echo "=========================================="

# Create labels for both subsets
for subset in "fit" "project"; do
    echo "  Creating ${subset} subset labels..."
    python3 << EOF
import pandas as pd

metadata = pd.read_csv("${AOU_METADATA}", sep='\t', low_memory=False)
metadata['sample_id'] = metadata['person_id'].astype(str)

fam = pd.read_csv("${DATA_DIR}/${subset}_subset.fam", sep=r'\s+', header=None, names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Phenotype'])
subset_samples = set(fam['IID'].astype(str))

labels = metadata[metadata['sample_id'].isin(subset_samples)].copy()
columns = ['sample_id']
for col in ['race_ethnicity', 'sex_at_birth', 'age']:
    if col in labels.columns: columns.append(col)

labels[columns].to_csv("${DATA_DIR}/${subset}_labels.csv", index=False)
EOF
done

echo "  ✓ Labels created"
echo ""
echo "Data Preparation Complete!"
echo "  Fit Subset: 60K White + All Non-White ($FIT_SAMPLES samples)"
echo "  Project Subset: Full Dataset ($PROJECT_SAMPLES samples)"