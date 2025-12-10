#!/bin/bash
#
# Process HGDP+1KGP data: create subsets and labels
#
# This script processes the downloaded raw data to create fit and transform subsets.
# Can run on COMPUTE NODE (no internet required).
#
# Smart caching: Skips processing if subsets already exist.
#
# Usage:
#   bash scripts/prepare_data.sh
#
# Prerequisites:
#   - Raw data downloaded (run scripts/download_data.sh first)
#   - plink2 available (from bin/plink2 or module/PATH)
#   - Virtual environment activated
#

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Paths
DATA_DIR="${SCRIPT_DIR}/data"
RAW_DIR="${DATA_DIR}/raw"

# ANSI colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
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
    echo -e "${RED}ERROR:${NC} $1"
}

# Print header
echo ""
echo "========================================="
echo "  HGDP+1KGP Data Processing"
echo "========================================="
echo ""

# Step 1: Check if processed data already exists
print_status "Checking for existing processed data..."

FIT_BED="${DATA_DIR}/fit_subset.bed"
TRANSFORM_BED="${DATA_DIR}/transform_subset.bed"
FIT_LABELS="${DATA_DIR}/hgdp_fit_labels.csv"
TRANSFORM_LABELS="${DATA_DIR}/hgdp_transform_labels.csv"
FIT_GEOGRAPHIC="${DATA_DIR}/hgdp_fit_geographic.csv"
TRANSFORM_GEOGRAPHIC="${DATA_DIR}/hgdp_transform_geographic.csv"
COLORMAP="${DATA_DIR}/colormap.json"

if [[ -f "${FIT_BED}" ]] && [[ -f "${TRANSFORM_BED}" ]] && \
   [[ -f "${FIT_LABELS}" ]] && [[ -f "${TRANSFORM_LABELS}" ]] && \
   [[ -f "${FIT_GEOGRAPHIC}" ]] && [[ -f "${TRANSFORM_GEOGRAPHIC}" ]] && \
   [[ -f "${COLORMAP}" ]]; then
    print_success "Data already processed!"
    echo ""
    echo "Found:"
    echo "  - fit_subset.{bed,bim,fam}"
    echo "  - transform_subset.{bed,bim,fam}"
    echo "  - hgdp_fit_labels.csv"
    echo "  - hgdp_transform_labels.csv"
    echo "  - hgdp_fit_geographic.csv"
    echo "  - hgdp_transform_geographic.csv"
    echo "  - colormap.json"
    echo ""
    echo "You can run: python examples/hgdp_1kgp/run_pipeline.py"
    exit 0
fi

print_warning "Processed data not found, will create subsets..."

# Step 2: Check if raw data exists
print_status "Checking for raw data..."

RAW_BED="${RAW_DIR}/full_dataset.bed"
RAW_METADATA="${RAW_DIR}/metadata.csv"

if [[ ! -f "${RAW_BED}" ]] || [[ ! -f "${RAW_METADATA}" ]]; then
    print_error "Raw data not found!"
    echo ""
    echo "Missing files in ${RAW_DIR}/"
    echo ""
    echo "Please run the download script first:"
    echo "  bash examples/hgdp_1kgp/download_data.sh"
    echo ""
    exit 1
fi

print_success "Raw data found"

# Step 3: Find plink2
print_status "Looking for plink2..."

PLINK2=""

# Check bin/ directory first (preferred)
if [[ -f "${PROJECT_ROOT}/bin/plink2" ]]; then
    PLINK2="${PROJECT_ROOT}/bin/plink2"
    print_success "Using plink2 from bin/plink2"
# Check if plink2 module is loaded
elif module list 2>&1 | grep -q plink2; then
    PLINK2="plink2"
    print_success "Using plink2 from loaded module"
# Check PATH
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

# Verify plink2 works
if ! ${PLINK2} --version &> /dev/null; then
    print_error "plink2 found but not executable!"
    exit 1
fi

# Step 4: Check for Python and required packages
print_status "Checking Python environment..."

if ! command -v python &> /dev/null; then
    print_error "Python not found!"
    echo ""
    echo "Please activate the virtual environment:"
    echo "  source .venv/bin/activate"
    echo ""
    exit 1
fi

# Check for pandas (required for processing metadata)
if ! python -c "import pandas" &> /dev/null; then
    print_error "pandas not installed!"
    echo ""
    echo "Please activate the virtual environment:"
    echo "  source .venv/bin/activate"
    echo ""
    exit 1
fi

print_success "Python environment ready"

# Step 5: Create fit indices (3,400 unrelated samples)
print_status "Creating fit indices (unrelated samples)..."

python3 - <<'PYEOF'
import pandas as pd
from pathlib import Path

# Read metadata
metadata_path = Path("examples/hgdp_1kgp/data/raw/metadata.csv")
metadata = pd.read_csv(metadata_path)

# Filter for unrelated samples
# filter_king_related == False means NOT related (i.e., unrelated)
unrelated = metadata[metadata['filter_king_related'] == False].copy()

# Create FID IID format for plink2 --keep
# Use project_meta.sample_id for both FID and IID
keep_df = pd.DataFrame({
    'FID': unrelated['project_meta.sample_id'],
    'IID': unrelated['project_meta.sample_id']
})

# Write indices
output_path = Path("examples/hgdp_1kgp/data/fit_indices.txt")
keep_df.to_csv(output_path, sep='\t', header=False, index=False)

print(f"Created fit indices: {len(keep_df)} samples")
PYEOF

if [[ ! -f "${DATA_DIR}/fit_indices.txt" ]]; then
    print_error "Failed to create fit indices!"
    exit 1
fi

# Count samples
FIT_COUNT=$(wc -l < "${DATA_DIR}/fit_indices.txt")
print_success "Created fit_indices.txt (${FIT_COUNT} samples)"

# Step 6: Create transform indices (4,094 all QC-passing samples)
print_status "Creating transform indices (all QC-passing samples)..."

python3 - <<'PYEOF'
import pandas as pd
from pathlib import Path

# Read metadata
metadata_path = Path("examples/hgdp_1kgp/data/raw/metadata.csv")
metadata = pd.read_csv(metadata_path)

# Filter for QC-passing samples
qc_pass = metadata[
    (metadata['filter_pca_outlier'] == False) &
    (metadata['hard_filtered'] == False) &
    (metadata['filter_contaminated'] == False)
].copy()

# Create FID IID format
keep_df = pd.DataFrame({
    'FID': qc_pass['project_meta.sample_id'],
    'IID': qc_pass['project_meta.sample_id']
})

# Write indices
output_path = Path("examples/hgdp_1kgp/data/transform_indices.txt")
keep_df.to_csv(output_path, sep='\t', header=False, index=False)

print(f"Created transform indices: {len(keep_df)} samples")
PYEOF

if [[ ! -f "${DATA_DIR}/transform_indices.txt" ]]; then
    print_error "Failed to create transform indices!"
    exit 1
fi

# Count samples
TRANSFORM_COUNT=$(wc -l < "${DATA_DIR}/transform_indices.txt")
print_success "Created transform_indices.txt (${TRANSFORM_COUNT} samples)"

# Step 7: Create PLINK subsets
print_status "Creating fit subset with plink2..."

${PLINK2} \
    --bfile "${RAW_DIR}/full_dataset" \
    --keep "${DATA_DIR}/fit_indices.txt" \
    --make-bed \
    --out "${DATA_DIR}/fit_subset" \
    --silent

if [[ -f "${FIT_BED}" ]]; then
    print_success "Created fit_subset.{bed,bim,fam}"
else
    print_error "Failed to create fit subset!"
    exit 1
fi

print_status "Creating transform subset with plink2..."

${PLINK2} \
    --bfile "${RAW_DIR}/full_dataset" \
    --keep "${DATA_DIR}/transform_indices.txt" \
    --make-bed \
    --out "${DATA_DIR}/transform_subset" \
    --silent

if [[ -f "${TRANSFORM_BED}" ]]; then
    print_success "Created transform_subset.{bed,bim,fam}"
else
    print_error "Failed to create transform subset!"
    exit 1
fi

# Step 8: Create label CSVs from metadata
print_status "Creating label CSVs..."

python3 - <<'PYEOF'
import pandas as pd
from pathlib import Path

# Read metadata
metadata_path = Path("examples/hgdp_1kgp/data/raw/metadata.csv")
metadata = pd.read_csv(metadata_path)

# Read fit indices
fit_indices_path = Path("examples/hgdp_1kgp/data/fit_indices.txt")
fit_indices = pd.read_csv(fit_indices_path, sep='\t', header=None, names=['FID', 'IID'])
fit_samples = set(fit_indices['IID'])

# Read transform indices
transform_indices_path = Path("examples/hgdp_1kgp/data/transform_indices.txt")
transform_indices = pd.read_csv(transform_indices_path, sep='\t', header=None, names=['FID', 'IID'])
transform_samples = set(transform_indices['IID'])

# Create fit labels
fit_labels = metadata[metadata['project_meta.sample_id'].isin(fit_samples)][
    ['project_meta.sample_id', 'Population', 'Genetic_region_merged']
].copy()
# Rename column to match expected format
fit_labels = fit_labels.rename(columns={'project_meta.sample_id': 'sample_id'})

fit_output = Path("examples/hgdp_1kgp/data/hgdp_fit_labels.csv")
fit_labels.to_csv(fit_output, index=False)
print(f"Created hgdp_fit_labels.csv ({len(fit_labels)} samples)")

# Create transform labels
transform_labels = metadata[metadata['project_meta.sample_id'].isin(transform_samples)][
    ['project_meta.sample_id', 'Population', 'Genetic_region_merged']
].copy()
# Rename column to match expected format
transform_labels = transform_labels.rename(columns={'project_meta.sample_id': 'sample_id'})

transform_output = Path("examples/hgdp_1kgp/data/hgdp_transform_labels.csv")
transform_labels.to_csv(transform_output, index=False)
print(f"Created hgdp_transform_labels.csv ({len(transform_labels)} samples)")
PYEOF

if [[ -f "${FIT_LABELS}" ]] && [[ -f "${TRANSFORM_LABELS}" ]]; then
    print_success "Created label CSVs"
else
    print_error "Failed to create label CSVs!"
    exit 1
fi

# Step 9: Create geographic coordinate CSVs (excluding Americas + specific populations)
print_status "Creating geographic coordinate CSVs..."

python3 - <<'PYEOF'
import pandas as pd
import numpy as np
from pathlib import Path

# Read metadata
metadata_path = Path("examples/hgdp_1kgp/data/raw/metadata.csv")
metadata = pd.read_csv(metadata_path)

# Read fit and transform indices
fit_indices_path = Path("examples/hgdp_1kgp/data/fit_indices.txt")
fit_indices = pd.read_csv(fit_indices_path, sep='\t', header=None, names=['FID', 'IID'])
fit_samples = set(fit_indices['IID'])

transform_indices_path = Path("examples/hgdp_1kgp/data/transform_indices.txt")
transform_indices = pd.read_csv(transform_indices_path, sep='\t', header=None, names=['FID', 'IID'])
transform_samples = set(transform_indices['IID'])

def extract_geographic_preservation_indices(metadata_subset):
    """
    Extract indices of samples that we expect to preserve geography.
    Excludes America region and specific populations (ACB, ASW, CEU).
    """
    # Exclude Americas
    american_idx = metadata_subset['Genetic_region_merged'] == 'America'

    # Exclude specific populations that don't preserve geography well
    excluded_pops_idx = metadata_subset['Population'].isin(['ACB', 'ASW', 'CEU'])

    # Return samples that are NOT in excluded categories
    return ~(american_idx | excluded_pops_idx)

# Process fit subset
fit_metadata = metadata[metadata['project_meta.sample_id'].isin(fit_samples)].copy()
fit_geo_mask = extract_geographic_preservation_indices(fit_metadata)
fit_geo_samples = fit_metadata[fit_geo_mask]

# Check for required geographic columns
geo_columns = ['latitude', 'longitude']
missing_cols = [col for col in geo_columns if col not in fit_geo_samples.columns]

if missing_cols:
    print(f"Warning: Missing geographic columns: {missing_cols}")
    print("Available columns:", list(fit_geo_samples.columns))
    # Try alternative column names
    alt_names = {
        'latitude': ['lat', 'Latitude', 'LAT'],
        'longitude': ['lon', 'lng', 'Longitude', 'LON', 'LNG']
    }

    for col in missing_cols:
        for alt in alt_names.get(col, []):
            if alt in fit_geo_samples.columns:
                print(f"Using '{alt}' for '{col}'")
                fit_geo_samples = fit_geo_samples.rename(columns={alt: col})
                break
        else:
            print(f"Error: No geographic column found for '{col}'")
            exit(1)

# Create fit geographic CSV
fit_geo_data = fit_geo_samples[['project_meta.sample_id', 'latitude', 'longitude']].copy()
fit_geo_data = fit_geo_data.rename(columns={'project_meta.sample_id': 'sample_id'})

# Remove rows with missing coordinates
fit_geo_data = fit_geo_data.dropna(subset=['latitude', 'longitude'])

fit_geo_output = Path("examples/hgdp_1kgp/data/hgdp_fit_geographic.csv")
fit_geo_data.to_csv(fit_geo_output, index=False)
print(f"Created hgdp_fit_geographic.csv ({len(fit_geo_data)} samples)")

# Process transform subset
transform_metadata = metadata[metadata['project_meta.sample_id'].isin(transform_samples)].copy()
transform_geo_mask = extract_geographic_preservation_indices(transform_metadata)
transform_geo_samples = transform_metadata[transform_geo_mask]

# Create transform geographic CSV
transform_geo_data = transform_geo_samples[['project_meta.sample_id', 'latitude', 'longitude']].copy()
transform_geo_data = transform_geo_data.rename(columns={'project_meta.sample_id': 'sample_id'})

# Remove rows with missing coordinates
transform_geo_data = transform_geo_data.dropna(subset=['latitude', 'longitude'])

transform_geo_output = Path("examples/hgdp_1kgp/data/hgdp_transform_geographic.csv")
transform_geo_data.to_csv(transform_geo_output, index=False)
print(f"Created hgdp_transform_geographic.csv ({len(transform_geo_data)} samples)")

# Print filtering summary
print(f"Fit subset (unrelated, QC-pass): {len(fit_metadata)} samples")
print(f"  - Geographic analysis: {len(fit_geo_data)} samples (excluded {len(fit_metadata) - len(fit_geo_data)} Americas/ACB/ASW/CEU)")

print(f"Transform subset (all QC-pass): {len(transform_metadata)} samples")
print(f"  - Geographic analysis: {len(transform_geo_data)} samples (excluded {len(transform_metadata) - len(transform_geo_data)} Americas/ACB/ASW/CEU)")
PYEOF

if [[ -f "${FIT_GEOGRAPHIC}" ]] && [[ -f "${TRANSFORM_GEOGRAPHIC}" ]]; then
    print_success "Created geographic coordinate CSVs"
else
    print_error "Failed to create geographic coordinate CSVs!"
    exit 1
fi

# Step 10: Copy colormap
print_status "Copying colormap..."

if [[ -f "${RAW_DIR}/colormap.json" ]]; then
    cp "${RAW_DIR}/colormap.json" "${DATA_DIR}/colormap.json"
    print_success "Copied colormap.json"
else
    print_warning "colormap.json not found in raw/, skipping..."
fi

# Step 10: Print success summary
echo ""
echo "========================================="
print_success "Data processing complete!"
echo "========================================="
echo ""
echo "Files created in ${DATA_DIR}:"
echo "  - fit_subset.{bed,bim,fam} (${FIT_COUNT} samples)"
echo "  - transform_subset.{bed,bim,fam} (${TRANSFORM_COUNT} samples)"
echo "  - hgdp_fit_labels.csv"
echo "  - hgdp_transform_labels.csv"
echo "  - hgdp_fit_geographic.csv (geographic coordinates, Americas excluded)"
echo "  - hgdp_transform_geographic.csv (geographic coordinates, Americas excluded)"
echo "  - colormap.json"
echo ""
echo "Next steps:"
echo "  python examples/hgdp_1kgp/run_pipeline.py"
echo ""
