#!/bin/bash
#
# UKBB-HGDP Cross-Projection Data Preparation
#
# This wrapper script:
# 1. Reads UKBB-specific paths from mappings_private.json
# 2. Calls the generic cross-projection script to create intersected PLINK files
# 3. Creates label CSV files from HGDP and UKBB metadata
#
# Usage:
#   bash /path/to/manifold_genetics/examples/ukbb/hgdp_1kgp_proj/prepare_data.sh
#

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Source common utilities
source "${PROJECT_ROOT}/examples/_shared/preprocessing/common.sh"

# Print header
print_header "UKBB-HGDP Cross-Projection Data Prep"

# Paths
DATA_DIR="${SCRIPT_DIR}/data"
MAPPINGS_FILE="${DATA_DIR}/mappings_private.json"
TEMP_DIR="${DATA_DIR}/temp"

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$TEMP_DIR"

# ============================================================================
# Step 1: Read mappings file
# ============================================================================
print_status "Reading data paths from mappings.json..."

if [[ ! -f "$MAPPINGS_FILE" ]]; then
    print_error "Mappings file not found: $MAPPINGS_FILE"
    echo "Please create mappings_private.json with data paths"
    exit 1
fi

# Extract paths using Python (resolve relative paths from PROJECT_ROOT)
python3 << EOF > "${TEMP_DIR}/paths.sh"
import json
from pathlib import Path

with open('$MAPPINGS_FILE') as f:
    mappings = json.load(f)

project_root = Path('$PROJECT_ROOT')

# Resolve paths (relative ones get resolved from project root)
for key, value in mappings.items():
    if value.startswith('/'):
        # Absolute path
        print(f"{key.upper()}='{value}'")
    else:
        # Relative path - resolve from project root
        resolved = project_root / value
        print(f"{key.upper()}='{resolved}'")
EOF

source "${TEMP_DIR}/paths.sh"

print_success "Loaded paths:"
echo "  UKBB PLINK: $UKBB_PLINK"
echo "  HGDP PLINK: $HGDP_PLINK"
echo "  Fit labels: $FIT_LABELS"
echo "  Project labels: $PROJECT_LABELS"

# ============================================================================
# Step 2: Call shared preprocessing script (intersection only)
# ============================================================================
print_status "Calling shared preprocessing script..."
echo ""

# Determine threads from SLURM or default
THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Call the shared script with skip flags (data already preprocessed, just do intersection)
bash "${PROJECT_ROOT}/examples/_shared/preprocessing/preprocess_cross_projection.sh" \
    --reference-plink "$HGDP_PLINK" \
    --biobank-plink "$UKBB_PLINK" \
    --output-dir "$DATA_DIR" \
    --temp-dir "$TEMP_DIR" \
    --memory 100000 \
    --threads "$THREADS" \
    --skip-wrayner --skip-giab --skip-hla --skip-ld-prune --skip-dedup --skip-maf

# ============================================================================
# Step 3: Filter labels to match intersected samples
# ============================================================================
print_status "Filtering labels to intersected samples..."

python3 << PYTHON_SCRIPT
import pandas as pd
import sys

# Read .fam files to get intersected sample IDs
print("  Reading intersected sample IDs...")
fit_fam = pd.read_csv('${DATA_DIR}/fit_subset.fam', sep=r'\s+', header=None,
                      names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Phenotype'])
project_fam = pd.read_csv('${DATA_DIR}/project_subset.fam', sep=r'\s+', header=None,
                          names=['FID', 'IID', 'PID', 'MID', 'Sex', 'Phenotype'])

fit_samples = set(fit_fam['IID'].astype(str))
project_samples = set(project_fam['IID'].astype(str))

print(f"    Fit samples: {len(fit_samples)}")
print(f"    Project samples: {len(project_samples)}")

# Filter fit labels (HGDP)
print("  Creating fit_labels.csv from source...")
fit_labels_source = pd.read_csv('${FIT_LABELS}')

# Handle flexible sample ID column names (e.g., 'project_meta.sample_id' or 'sample_id')
sample_id_col = None
for col in ['sample_id', 'project_meta.sample_id']:
    if col in fit_labels_source.columns:
        sample_id_col = col
        break

if sample_id_col is None:
    print(f"  Error: No sample ID column found in fit labels source")
    print(f"  Available columns: {', '.join(fit_labels_source.columns)}")
    sys.exit(1)

# Rename to 'sample_id' if needed
if sample_id_col != 'sample_id':
    fit_labels_source = fit_labels_source.rename(columns={sample_id_col: 'sample_id'})

fit_labels_source['sample_id'] = fit_labels_source['sample_id'].astype(str)
fit_labels = fit_labels_source[fit_labels_source['sample_id'].isin(fit_samples)].copy()
fit_labels.to_csv('${DATA_DIR}/fit_labels.csv', index=False)
print(f"  ✓ Created fit_labels.csv: {len(fit_labels)} samples, {len(fit_labels.columns)} columns")

# Filter project labels (UKBB)
print("  Creating project_labels.csv from source...")
project_labels_source = pd.read_csv('${PROJECT_LABELS}')

# Handle flexible sample ID column names
sample_id_col = None
for col in ['sample_id', 'project_meta.sample_id']:
    if col in project_labels_source.columns:
        sample_id_col = col
        break

if sample_id_col is None:
    print(f"  Error: No sample ID column found in project labels source")
    print(f"  Available columns: {', '.join(project_labels_source.columns)}")
    sys.exit(1)

# Rename to 'sample_id' if needed
if sample_id_col != 'sample_id':
    project_labels_source = project_labels_source.rename(columns={sample_id_col: 'sample_id'})

project_labels_source['sample_id'] = project_labels_source['sample_id'].astype(str)
project_labels = project_labels_source[project_labels_source['sample_id'].isin(project_samples)].copy()
project_labels.to_csv('${DATA_DIR}/project_labels.csv', index=False)
print(f"  ✓ Created project_labels.csv: {len(project_labels)} samples, {len(project_labels.columns)} columns")
PYTHON_SCRIPT

# Cleanup
rm -f "${TEMP_DIR}/paths.sh"

echo ""
print_success "UKBB-HGDP data preparation complete!"
echo ""
echo "Generated files:"
echo "  📊 Intersected PLINK files:"
echo "    - ${DATA_DIR}/fit_subset.{bed,bim,fam} (HGDP)"
echo "    - ${DATA_DIR}/project_subset.{bed,bim,fam} (UKBB)"
echo ""
echo "  🏷️ Label files:"
echo "    - ${DATA_DIR}/fit_labels.csv (HGDP labels)"
echo "    - ${DATA_DIR}/project_labels.csv (UKBB labels)"
echo ""
