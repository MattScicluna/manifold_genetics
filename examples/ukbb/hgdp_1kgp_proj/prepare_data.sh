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

# ANSI colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
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
    echo -e "${RED}✗${NC} $1"
}

# Print header
echo ""
echo "========================================="
echo "  UKBB-HGDP Cross-Projection Data Prep"
echo "========================================="
echo ""

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
print_status "Reading data paths from mappings_private.json..."

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
echo "  HGDP Metadata: $HGDP_METADATA"
if [[ -n "$UKBB_METADATA" ]]; then
    echo "  UKBB Metadata: $UKBB_METADATA"
fi

# ============================================================================
# Step 2: Call generic cross-projection prepare script
# ============================================================================
print_status "Calling generic cross-projection prepare script..."
echo ""

# Determine threads from SLURM or default
THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Call the generic script (creates intersected PLINK files only)
bash "${PROJECT_ROOT}/examples/generic/hgdp_1kgp_proj/prepare_data.sh" \
    --reference-plink "$HGDP_PLINK" \
    --biobank-plink "$UKBB_PLINK" \
    --output-dir "$DATA_DIR" \
    --temp-dir "$TEMP_DIR" \
    --memory 100000 \
    --threads "$THREADS"

# ============================================================================
# Step 3: Create label files from metadata
# ============================================================================
print_status "Creating label files from metadata..."

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

# Filter HGDP labels
print("  Creating fit_labels.csv from HGDP metadata...")
hgdp_labels = pd.read_csv('${HGDP_METADATA}')
if 'sample_id' not in hgdp_labels.columns:
    print("  Error: 'sample_id' column not found in HGDP metadata")
    sys.exit(1)

hgdp_labels['sample_id'] = hgdp_labels['sample_id'].astype(str)
fit_labels = hgdp_labels[hgdp_labels['sample_id'].isin(fit_samples)].copy()
fit_labels.to_csv('${DATA_DIR}/fit_labels.csv', index=False)
print(f"  ✓ Created fit_labels.csv: {len(fit_labels)} samples, {len(fit_labels.columns)} columns")
print(f"    Columns: {', '.join(fit_labels.columns[:5])}...")

# Create UKBB labels
print("  Creating project_labels.csv...")

# Check if ukbb_labels.csv exists (pre-created labels)
import os
if os.path.exists('${DATA_DIR}/ukbb_labels.csv'):
    print("    Using existing ukbb_labels.csv...")
    ukbb_labels = pd.read_csv('${DATA_DIR}/ukbb_labels.csv')
    if 'sample_id' not in ukbb_labels.columns:
        print("  Error: 'sample_id' column not found in ukbb_labels.csv")
        sys.exit(1)
    ukbb_labels['sample_id'] = ukbb_labels['sample_id'].astype(str)
    project_labels = ukbb_labels[ukbb_labels['sample_id'].isin(project_samples)].copy()
    project_labels.to_csv('${DATA_DIR}/project_labels.csv', index=False)
    print(f"  ✓ Created project_labels.csv: {len(project_labels)} samples, {len(project_labels.columns)} columns")
    print(f"    Columns: {', '.join(project_labels.columns[:5])}...")
else:
    print("  ⚠ ukbb_labels.csv not found, creating basic project_labels.csv with sample_id only")
    project_labels = pd.DataFrame({'sample_id': sorted(project_samples)})
    project_labels.to_csv('${DATA_DIR}/project_labels.csv', index=False)
    print(f"  ✓ Created project_labels.csv: {len(project_labels)} samples (sample_id only)")
    print("")
    print("  To add UKBB metadata columns:")
    print(f"    1. Create ${DATA_DIR}/ukbb_labels.csv with columns:")
    print("       sample_id,self_described_ancestry,Population,...")
    print("    2. Re-run this script")
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
