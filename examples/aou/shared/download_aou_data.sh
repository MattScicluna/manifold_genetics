#!/bin/bash
#
# Shared script for downloading All of Us genotype data
#
# This script:
# 1. Downloads AoU V8 PLINK files from GCS bucket
# 2. Fixes FAM file format (adds Family ID)
# 3. Splits by chromosome (1-22)
# 4. Optionally extracts metadata from BigQuery
#
# NOTE: Filtering (MAF, geno, indel removal) is done in prepare_data.sh
#       to allow each experiment to tune its own parameters.
#
# Usage:
#   bash examples/aou/shared/download_aou_data.sh [SKIP_METADATA]
#
# Arguments:
#   SKIP_METADATA: Optional flag to skip metadata extraction (pass "skip" to skip)
#
# Environment variables:
#   GOOGLE_PROJECT: GCS project ID (required for download)
#   WORKSPACE_CDR: CDR version for metadata extraction (optional)
#   SLURM_CPUS_PER_TASK: Number of CPU cores for parallel processing (default: 4)
#
# Data is downloaded to: examples/aou/shared/data/
# This allows multiple experiments to share the same downloaded data.
#

set -e

# Get script directory and set shared data location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/data"
SKIP_METADATA="${1:-}"

# Configuration
CPU_CORES="${SLURM_CPUS_PER_TASK:-4}"
AOU_BUCKET_ROOT_V8="gs://fc-aou-datasets-controlled/v8/microarray/plink"
GOOGLE_PROJECT="${GOOGLE_PROJECT:-}"
CDR_VERSION="${WORKSPACE_CDR:-}"

# Directories
AOU_DIR="${OUTPUT_DIR}/AllofUs_V8"
META_DIR="${OUTPUT_DIR}/Metadata"

# Create directories
mkdir -p "${AOU_DIR}" "${META_DIR}"

# ANSI colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo ""
echo "=========================================="
echo "  All of Us Data Download (V8)"
echo "=========================================="
echo ""
echo "Downloading to shared location: ${OUTPUT_DIR}"
echo "This data will be reused by all AoU experiments."
echo ""
echo "Configuration:"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  CPU cores: ${CPU_CORES}"
echo "  Google project: ${GOOGLE_PROJECT:-'Not set'}"
echo "  CDR version: ${CDR_VERSION:-'Not set'}"
echo "  Skip metadata: ${SKIP_METADATA:-'No'}"
echo ""

# =============================================================================
# STEP 1: All of Us Genotype Download (V8)
# =============================================================================
print_status "Downloading All of Us Genotype Data (V8)..."

# 1. Verify bucket access
if [[ -n "${GOOGLE_PROJECT}" ]]; then
    gsutil -u "${GOOGLE_PROJECT}" ls -lh "${AOU_BUCKET_ROOT_V8}/arrays.*" || print_warning "Could not list bucket"
else
    print_warning "GOOGLE_PROJECT not set, gsutil commands may fail"
fi

# 2. Download PLINK files
for ext in bed bim fam; do
    SRC="${AOU_BUCKET_ROOT_V8}/arrays.${ext}"
    DEST="${AOU_DIR}/extractedChrAll.${ext}"

    if [[ ! -f "${DEST}" ]]; then
        echo "Downloading ${ext} file..."
        if [[ -n "${GOOGLE_PROJECT}" ]]; then
            gsutil -u "${GOOGLE_PROJECT}" cp -r "${SRC}" "${DEST}"
        else
            echo "  ERROR: GOOGLE_PROJECT not set. Please set it:"
            echo "    export GOOGLE_PROJECT=your-project-id"
            exit 1
        fi
    else
        echo "  ${ext} file already exists"
    fi
done

# 3. Validate file sizes
print_status "Validating downloaded files..."
wc "${AOU_DIR}/extractedChrAll.bim"
wc "${AOU_DIR}/extractedChrAll.fam"

# 4. Fix FAM file (add Family ID)
FAM_PATH="${AOU_DIR}/extractedChrAll.fam"
ORIG_FAM_PATH="${AOU_DIR}/extractedChrAll.Original.fam"

if [[ ! -f "${ORIG_FAM_PATH}" ]]; then
    print_status "Fixing AoU FAM file (adding Family ID)..."
    mv "${FAM_PATH}" "${ORIG_FAM_PATH}"
    awk '{print "AOU\t"$2"\t"$3"\t"$4"\t"$5"\t-9"}' "${ORIG_FAM_PATH}" > "${FAM_PATH}"
else
    echo "  FAM file already fixed"
fi

# 5. Split by chromosome (1-22) with memory limit
print_status "Splitting AoU data by chromosome (using ${CPU_CORES} cores)..."
if [[ ! -f "${AOU_DIR}/extractedChr22.bed" ]]; then
    seq 1 22 | xargs -I {} -P "${CPU_CORES}" sh -c \
        "plink --bfile ${AOU_DIR}/extractedChrAll \
               --keep-allele-order --allow-no-sex --chr {} \
               --memory 8000 \
               --make-bed --out ${AOU_DIR}/extractedChr{}"
else
    echo "  AoU data already split by chromosome"
fi

print_success "AoU genotype data downloaded and split"
echo ""

# =============================================================================
# STEP 2: Metadata Extraction (Optional)
# =============================================================================
if [[ "${SKIP_METADATA}" == "skip" ]]; then
    print_status "Skipping metadata extraction (SKIP_METADATA flag set)"
else
    print_status "Extracting AoU Metadata..."

    if [[ -z "${CDR_VERSION}" ]]; then
        print_warning "CDR_VERSION (WORKSPACE_CDR) not set"
        echo "  Metadata queries will be skipped"
        echo "  To run metadata extraction, set:"
        echo "    export WORKSPACE_CDR=your-cdr-version"
    else
        echo "  CDR Version: ${CDR_VERSION}"
        echo ""
        echo "  Running BigQuery metadata extraction..."
        echo "  Note: This step requires Python with pandas-gbq"
        echo ""

        # Create Python script for metadata extraction
        cat > "${META_DIR}/extract_metadata.py" <<'EOF'
import os
import pandas as pd

CDR_VERSION = os.environ.get("WORKSPACE_CDR", "")
META_DIR = os.path.dirname(__file__)

print(f"CDR Version: {CDR_VERSION}")
print(f"Output directory: {META_DIR}")

# Query 1: Demographics
print("\nQuerying Demographics...")
sql_person = f"""
    SELECT
        person.person_id,
        person.gender_concept_id,
        p_gender_concept.concept_name as gender,
        person.birth_datetime as date_of_birth,
        person.race_concept_id,
        p_race_concept.concept_name as race,
        person.ethnicity_concept_id,
        p_ethnicity_concept.concept_name as ethnicity,
        person.sex_at_birth_concept_id,
        p_sex_at_birth_concept.concept_name as sex_at_birth
    FROM `{CDR_VERSION}.person` person
    LEFT JOIN `{CDR_VERSION}.concept` p_gender_concept
        ON person.gender_concept_id = p_gender_concept.concept_id
    LEFT JOIN `{CDR_VERSION}.concept` p_race_concept
        ON person.race_concept_id = p_race_concept.concept_id
    LEFT JOIN `{CDR_VERSION}.concept` p_ethnicity_concept
        ON person.ethnicity_concept_id = p_ethnicity_concept.concept_id
    LEFT JOIN `{CDR_VERSION}.concept` p_sex_at_birth_concept
        ON person.sex_at_birth_concept_id = p_sex_at_birth_concept.concept_id
"""
df_person = pd.read_gbq(sql_person, dialect="standard")
df_person.to_csv(os.path.join(META_DIR, 'DemographicData.tsv'), sep="\t", index=False)
print(f"  Saved {len(df_person)} demographic records")

# Query 2: Socioeconomics
print("\nQuerying Socioeconomics...")
sql_ses = f"""
    SELECT
        observation.person_id,
        zip_code.zip3_as_string as zip_code,
        zip_code.fraction_assisted_income as assisted_income,
        zip_code.fraction_high_school_edu as high_school_education,
        zip_code.median_income,
        zip_code.fraction_no_health_ins as no_health_insurance,
        zip_code.fraction_poverty as poverty,
        zip_code.deprivation_index
    FROM `{CDR_VERSION}.zip3_ses_map` zip_code
    JOIN `{CDR_VERSION}.observation` observation
        ON CAST(SUBSTR(observation.value_as_string, 0, STRPOS(observation.value_as_string, '*') - 1) AS INT64) = zip_code.zip3
        AND observation_source_concept_id = 1585250
        AND observation.value_as_string NOT LIKE 'Res%'
"""
df_ses = pd.read_gbq(sql_ses, dialect="standard")
df_ses.to_csv(os.path.join(META_DIR, 'SocioeconomicZipCodes.tsv'), sep="\t", index=False)
print(f"  Saved {len(df_ses)} socioeconomic records")

print("\nMetadata extraction complete!")
EOF

        # Run metadata extraction if pandas-gbq is available
        if command -v python &> /dev/null; then
            python "${META_DIR}/extract_metadata.py" || print_warning "Metadata extraction failed (pandas-gbq may not be installed)"
        else
            print_warning "Python not found, skipping metadata extraction"
        fi
    fi
fi

echo ""

# =============================================================================
# Summary
# =============================================================================
print_success "All of Us data download complete!"
echo ""
echo "Output files:"
echo "  - ${AOU_DIR}/extractedChrAll.{bed,bim,fam}  (full dataset, ~1.7M SNPs)"
echo "  - ${AOU_DIR}/extractedChr{1..22}.{bed,bim,fam}  (split by chromosome)"
if [[ "${SKIP_METADATA}" != "skip" ]] && [[ -n "${CDR_VERSION}" ]]; then
    echo "  - ${META_DIR}/DemographicData.tsv  (demographics)"
    echo "  - ${META_DIR}/SocioeconomicZipCodes.tsv  (socioeconomics)"
fi
echo ""
echo "Next step: Run prepare_data.sh in your experiment directory"
echo "  This will filter both AoU and reference data with experiment-specific parameters."
echo ""
