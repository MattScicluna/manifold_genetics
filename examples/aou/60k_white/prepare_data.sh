#!/bin/bash
#
# Prepare AoU 60K White Subset Data
# This script integrates the aou_pipeline steps to:
# 1. Download/prepare AoU genotype data (step3)
# 2. Query AoU metadata (step4)
# 3. Filter for white/European ancestry samples
# 4. Randomly select 60K for fit subset
# 5. Use remaining for project subset
# 6. Create labels and colormaps
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TEMP_DIR="${DATA_DIR}/temp"

# Configuration
CPU_CORES="${SLURM_CPUS_PER_TASK:-4}"
GOOGLE_PROJECT="${GOOGLE_PROJECT:-}"
CDR_VERSION="${WORKSPACE_CDR:-}"

# Shared AoU data directories
SHARED_AOU_DIR="${SCRIPT_DIR}/../shared/data/AllofUs_V8"
SHARED_META_DIR="${SCRIPT_DIR}/../shared/data/Metadata"

# Use intersected AoU data from hgdp_1kgp_proj (much fewer SNPs!)
INTERSECTED_AOU="${SCRIPT_DIR}/../hgdp_1kgp_proj/data/project_subset"

echo "=========================================="
echo "  AoU 60K White Subset Data Preparation"
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

# Check if intersected AoU data exists (from hgdp_1kgp_proj)
if [[ ! -f "${INTERSECTED_AOU}.bed" ]]; then
    echo "  ✗ Intersected AoU data not found!"
    echo ""
    echo "This pipeline uses the HGDP-intersected AoU dataset to reduce SNP count"
    echo "and memory requirements (~500K SNPs instead of 1.7M SNPs)."
    echo ""
    echo "Please run the hgdp_1kgp_proj pipeline first:"
    echo "  cd ${SCRIPT_DIR}/../hgdp_1kgp_proj"
    echo "  bash prepare_data.sh"
    echo ""
    echo "This will create: ${INTERSECTED_AOU}.{bed,bim,fam}"
    exit 1
fi

# Get stats from intersected data
AOU_SAMPLES=$(wc -l < "${INTERSECTED_AOU}.fam")
COMMON_SNPS=$(wc -l < "${INTERSECTED_AOU}.bim")

echo "  ✓ Found intersected AoU data:"
echo "    Samples: $AOU_SAMPLES"
echo "    SNPs: $COMMON_SNPS (intersected with HGDP+1KGP)"
echo ""

# =============================================================================
# STEP 2: Filter for White/European Ancestry
# =============================================================================
echo "=========================================="
echo "Step 2: Filter for White/European Ancestry"
echo "=========================================="

# Check if AoU metadata exists
AOU_METADATA="${SHARED_META_DIR}/DemographicData.tsv"

if [[ ! -f "$AOU_METADATA" ]]; then
    echo "  ✗ AoU metadata not found at: $AOU_METADATA"
    echo ""
    echo "Please ensure demographic data is available."
    exit 1
fi

# Filter for white/European vs Other samples using Python
echo "  Classifying samples..."

python3 << EOF
import pandas as pd

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

# Filter for white/European ancestry
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

print(f"    White/European samples: {len(white_samples)}")
print(f"    Other samples: {len(other_samples)}")

# Write sample lists
for name, df in [("white", white_samples), ("other", other_samples)]:
    with open(f"${DATA_DIR}/{name}_samples.txt", 'w') as f:
        for sample_id in df['sample_id']:
            if sample_id in iid_to_fid:
                f.write(f"{iid_to_fid[sample_id]}\t{sample_id}\n")

print(f"  ✓ Saved sample lists to ${DATA_DIR}/")
EOF

WHITE_SAMPLE_COUNT=$(wc -l < "${DATA_DIR}/white_samples.txt")
OTHER_SAMPLE_COUNT=$(wc -l < "${DATA_DIR}/other_samples.txt")
echo "  White/European samples: $WHITE_SAMPLE_COUNT"
echo "  Other samples: $OTHER_SAMPLE_COUNT"
echo ""

# Find plink2
echo "Looking for plink2..."
PLINK2=""

if [[ -f "${PROJECT_ROOT}/bin/plink2" ]]; then
    PLINK2="${PROJECT_ROOT}/bin/plink2"
    echo "  ✓ Using plink2 from bin/plink2"
elif command -v plink2 &> /dev/null; then
    PLINK2="plink2"
    echo "  ✓ Using plink2 from PATH"
else
    echo "  ✗ plink2 not found!"
    exit 1
fi

# Extract subsets from intersected AoU data
for group in "white" "other"; do
    if [[ ! -f "${DATA_DIR}/${group}_all.bed" ]]; then
        echo "  Extracting ${group} samples from intersected AoU data..."
        ${PLINK2} --bfile ${INTERSECTED_AOU} \
            --keep ${DATA_DIR}/${group}_samples.txt \
            --memory 100000 \
            --make-bed \
            --out ${DATA_DIR}/${group}_all
    else
        echo "  ${group} samples dataset already exists"
    fi
done

WHITE_ALL_SAMPLES=$(wc -l < "${DATA_DIR}/white_all.fam")
OTHER_ALL_SAMPLES=$(wc -l < "${DATA_DIR}/other_all.fam")
echo "  ✓ Base subsets created"
echo ""

# =============================================================================
# STEP 3: Create Fit Subset (60K White + All Other)
# =============================================================================
echo "=========================================="
echo "Step 3: Create Fit Subset (60K White + All Other)"
echo "=========================================="

# 1. Subsample White group
TARGET_WHITE_FIT=60000
if [[ ! -f "${DATA_DIR}/white_fit.bed" ]]; then
    echo "  Subsampling $TARGET_WHITE_FIT white samples..."
    if [[ $WHITE_ALL_SAMPLES -le $TARGET_WHITE_FIT ]]; then
        cp "${DATA_DIR}/white_all.bed" "${DATA_DIR}/white_fit.bed"
        cp "${DATA_DIR}/white_all.bim" "${DATA_DIR}/white_fit.bim"
        cp "${DATA_DIR}/white_all.fam" "${DATA_DIR}/white_fit.fam"
    else
        ${PLINK2} --bfile ${DATA_DIR}/white_all \
            --thin-indiv-count $TARGET_WHITE_FIT \
            --seed 42 \
            --memory 100000 \
            --make-bed \
            --out ${DATA_DIR}/white_fit
    fi
fi

# 2. Merge White Fit + All Other
if [[ ! -f "${DATA_DIR}/fit_subset.bed" ]]; then
    echo "  Merging 60K white samples with all other samples..."
    echo "${DATA_DIR}/other_all" > "${TEMP_DIR}/merge_list.txt"
    ${PLINK2} --bfile ${DATA_DIR}/white_fit \
        --pmerge ${DATA_DIR}/other_all \
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

# Create fit labels
echo "  Creating fit subset labels..."

python3 << EOF
import pandas as pd

# Read metadata
metadata = pd.read_csv("${AOU_METADATA}", sep='\t', low_memory=False)
metadata['sample_id'] = metadata['person_id'].astype(str)

# Read fit .fam file
fam = pd.read_csv("${DATA_DIR}/fit_subset.fam", sep=r'\s+', header=None,
                  names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Phenotype'])
fit_samples = set(fam['IID'].astype(str))

# Filter metadata for fit samples
fit_labels = metadata[metadata['sample_id'].isin(fit_samples)].copy()

# Select relevant columns
columns_to_keep = ['sample_id']
if 'race_ethnicity' in fit_labels.columns:
    columns_to_keep.append('race_ethnicity')
if 'sex_at_birth' in fit_labels.columns:
    columns_to_keep.append('sex_at_birth')
if 'age' in fit_labels.columns:
    columns_to_keep.append('age')

fit_labels = fit_labels[columns_to_keep]

# Save labels
fit_labels.to_csv("${DATA_DIR}/fit_labels.csv", index=False)
print(f"  ✓ Saved fit labels: {len(fit_labels)} samples")
EOF

# Create project labels (if project subset exists)
if [[ -f "${DATA_DIR}/project_subset.bed" ]]; then
    echo "  Creating project subset labels..."

    python3 << EOF
import pandas as pd

# Read metadata
metadata = pd.read_csv("${AOU_METADATA}", sep='\t', low_memory=False)
metadata['sample_id'] = metadata['person_id'].astype(str)

# Read project .fam file
fam = pd.read_csv("${DATA_DIR}/project_subset.fam", sep=r'\s+', header=None,
                  names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Phenotype'])
project_samples = set(fam['IID'].astype(str))

# Filter metadata for project samples
project_labels = metadata[metadata['sample_id'].isin(project_samples)].copy()

# Select relevant columns
columns_to_keep = ['sample_id']
if 'race_ethnicity' in project_labels.columns:
    columns_to_keep.append('race_ethnicity')
if 'sex_at_birth' in project_labels.columns:
    columns_to_keep.append('sex_at_birth')
if 'age' in project_labels.columns:
    columns_to_keep.append('age')

project_labels = project_labels[columns_to_keep]

# Save labels
project_labels.to_csv("${DATA_DIR}/project_labels.csv", index=False)
print(f"  ✓ Saved project labels: {len(project_labels)} samples")
EOF
fi

echo "  ✓ Label files created"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=========================================="
echo "Data Preparation Complete!"
echo "=========================================="
echo ""
echo "✓ All steps completed successfully:"
echo "  ✓ Step 1: Used HGDP-intersected AoU data ($COMMON_SNPS SNPs)"
echo "  ✓ Step 2: Filtered for White ($WHITE_ALL_SAMPLES) and Other ($OTHER_ALL_SAMPLES) samples"
echo "  ✓ Step 3: Created fit subset ($FIT_SAMPLES samples) - 60K White + All Other"
echo "  ✓ Step 4: Created project subset ($PROJECT_SAMPLES samples) - Entire Dataset"
echo "  ✓ Step 5: Created label files"
echo ""
echo "Generated files in ${DATA_DIR}:"
echo "  📊 Processed PLINK data:"
echo "    - fit_subset.{bed,bim,fam}     (Fit subset, $FIT_SAMPLES samples)"
echo "    - project_subset.{bed,bim,fam} (Project subset, $PROJECT_SAMPLES samples)"
echo ""
echo "  🏷️  Sample labels:"
echo "    - fit_labels.csv"
echo "    - project_labels.csv"
echo ""
echo "Next step: Run the pipeline analysis"
echo "  bash run_pipeline.sh"
echo ""
