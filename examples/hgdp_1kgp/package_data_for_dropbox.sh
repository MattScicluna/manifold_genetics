#!/bin/bash
#
# Package HGDP+1KGP data for Dropbox upload
#
# This script packages the FULL dataset (4,151 samples) along with metadata
# and colormap for distribution. Users will download this and run prepare_data.sh
# to create fit/transform subsets locally.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==========================================="
echo "Package HGDP+1KGP Data for Dropbox"
echo "==========================================="
echo ""

# Source data location
SOURCE_DIR="/lustre06/project/6065672/sciclun4/ActiveProjects/manyGenomes/data_new/hgdp"
GENOTYPES_DIR="${SOURCE_DIR}/standalone/genotypes"
METADATA_FILE="/lustre06/project/6065672/sciclun4/ActiveProjects/manyGenomes/data/hgdp+1kgp/gnomad_derived_metadata_with_filtered_sampleids.csv"

# Output location
PACKAGE_DIR="${PROJECT_ROOT}/data_package"
OUTPUT_ARCHIVE="${PACKAGE_DIR}/hgdp_1kgp_full.tar.gz"

# Filenames (shortened for convenience)
FULL_PLINK_BASE="gnomad.genomes.v3.1.2.hgdp_tgp.PASSfiltered.newIDs.onlySNPs.noDuplicatePos.noMiss5perc.maf0.01.LDpruned_500_50_0.05.woLowComplexity_noHLA"

echo "Step 1: Verify source files exist"
echo "----------------------------------"

# Check PLINK files
for ext in bed bim fam; do
    file="${GENOTYPES_DIR}/${FULL_PLINK_BASE}.${ext}"
    if [ ! -f "$file" ]; then
        echo "ERROR: Missing file: $file"
        exit 1
    fi
    echo "✓ Found: ${FULL_PLINK_BASE}.${ext}"
done

# Check metadata
if [ ! -f "$METADATA_FILE" ]; then
    echo "ERROR: Missing metadata file: $METADATA_FILE"
    exit 1
fi
echo "✓ Found: metadata CSV"

# Check colormap
if [ ! -f "${SOURCE_DIR}/colormap.json" ]; then
    echo "ERROR: Missing colormap: ${SOURCE_DIR}/colormap.json"
    exit 1
fi
echo "✓ Found: colormap.json"

echo ""
echo "Step 2: Create staging directory"
echo "----------------------------------"

# Create clean package directory
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"
echo "✓ Created: $PACKAGE_DIR"

echo ""
echo "Step 3: Copy files to staging"
echo "----------------------------------"

# Copy and rename PLINK files to simpler names
echo "Copying PLINK files (this may take a moment)..."
cp "${GENOTYPES_DIR}/${FULL_PLINK_BASE}.bed" "${PACKAGE_DIR}/full_dataset.bed"
cp "${GENOTYPES_DIR}/${FULL_PLINK_BASE}.bim" "${PACKAGE_DIR}/full_dataset.bim"
cp "${GENOTYPES_DIR}/${FULL_PLINK_BASE}.fam" "${PACKAGE_DIR}/full_dataset.fam"
echo "✓ Copied: full_dataset.{bed,bim,fam}"

# Copy metadata
cp "$METADATA_FILE" "${PACKAGE_DIR}/metadata.csv"
echo "✓ Copied: metadata.csv"

# Copy colormap
cp "${SOURCE_DIR}/colormap.json" "${PACKAGE_DIR}/colormap.json"
echo "✓ Copied: colormap.json"

echo ""
echo "Step 4: Create README"
echo "----------------------------------"

cat > "${PACKAGE_DIR}/README.txt" << 'EOF'
# HGDP+1KGP Test Data for manifold-genetics

## Contents

This archive contains the full HGDP+1000 Genomes Project dataset used for
testing the manifold-genetics package.

**Files**:
- `full_dataset.bed` - PLINK binary genotype data (4,151 samples, 172,152 SNPs)
- `full_dataset.bim` - PLINK variant information
- `full_dataset.fam` - PLINK sample information
- `metadata.csv` - Sample metadata with population labels and QC flags
- `colormap.json` - Color mapping for population visualization
- `README.txt` - This file

## Data Processing

**Source**: gnomAD v3.1.2 HGDP+1KGP joint call set

**Filtering applied**:
- PASS filters only (gnomAD v3.1.2)
- SNPs only (no indels or complex variants)
- ≤5% missingness per variant
- MAF ≥ 0.01
- LD pruning: 500kb windows, r² ≤ 0.05, step size 50 variants
- Low complexity regions excluded
- HLA region excluded (chr6:28477797-33448354, GRCh38)

**Samples**:
- Total: 4,151 individuals (HGDP ~929, 1KGP ~3,165)
- Unrelated: 3,400 individuals
- Related: 751 individuals
- 7 genetic regions: Africa, Americas, Central/South Asia, East Asia, Europe, Middle East, Oceania

**Variants**: 172,152 approximately independent SNPs

## Usage

After downloading this archive, run the data preparation script to create
analysis-ready subsets:

1. Extract the archive (done automatically by download_data.sh)
2. Run `bash scripts/prepare_data.sh` to create:
   - fit_subset (3,400 unrelated samples for training)
   - transform_subset (4,094 QC-passing samples for testing)
   - Label CSV files
   - Colormap

See the manifold-genetics README for detailed instructions.

## File Formats

**PLINK binary format** (.bed/.bim/.fam):
- Standard PLINK 1.9/2.0 format
- bed: binary genotype matrix
- bim: variant information (chr, ID, position, alleles)
- fam: sample information (family ID, individual ID)

**Metadata CSV**:
- Sample-level metadata from gnomAD
- Key columns:
  - `project_meta.sample_id`: Sample identifier
  - `Related`: 'Unrelated' or 'Related'
  - `Population`: Fine-grained population label
  - `Genetic_region_merged`: Coarse genetic region
  - `filter_pca_outlier`, `hard_filtered`, `filter_contaminated`: QC flags

**Colormap JSON**:
- Maps population labels to hex colors for plotting
- Three label types: Population, Genetic_region_merged, self_described_ancestry

## References

- gnomAD: https://gnomad.broadinstitute.org/
- HGDP: https://www.hagsc.org/hgdp/
- 1000 Genomes: https://www.internationalgenome.org/

## License

Data derived from gnomAD v3.1.2 (ODbL 1.0 license) and 1000 Genomes Project.
See individual project websites for licensing details.

EOF

echo "✓ Created: README.txt"

echo ""
echo "Step 5: Create archive"
echo "----------------------------------"

echo "Creating tar.gz archive (this may take a moment)..."
cd "$PACKAGE_DIR"
tar -czf "hgdp_1kgp_full.tar.gz" \
    full_dataset.bed \
    full_dataset.bim \
    full_dataset.fam \
    metadata.csv \
    colormap.json \
    README.txt

echo "✓ Created: hgdp_1kgp_full.tar.gz"

echo ""
echo "Step 6: Verify archive"
echo "----------------------------------"

# Get file size
FILE_SIZE=$(du -h "${OUTPUT_ARCHIVE}" | awk '{print $1}')
echo "Archive size: ${FILE_SIZE}"

# Calculate MD5 checksum
echo "Calculating MD5 checksum..."
if command -v md5sum &> /dev/null; then
    MD5=$(md5sum "${OUTPUT_ARCHIVE}" | awk '{print $1}')
elif command -v md5 &> /dev/null; then
    MD5=$(md5 -q "${OUTPUT_ARCHIVE}")
else
    MD5="(md5sum not available)"
fi
echo "MD5 checksum: ${MD5}"

# Test extraction
echo "Testing archive integrity..."
tar -tzf "${OUTPUT_ARCHIVE}" > /dev/null 2>&1
echo "✓ Archive integrity verified"

# List contents
echo ""
echo "Archive contents:"
tar -tzf "${OUTPUT_ARCHIVE}"

echo ""
echo "==========================================="
echo "Packaging Complete!"
echo "==========================================="
echo ""
echo "Output file:"
echo "  ${OUTPUT_ARCHIVE}"
echo ""
echo "File size: ${FILE_SIZE}"
echo "MD5: ${MD5}"
echo ""
echo "Next steps:"
echo "1. Upload ${OUTPUT_ARCHIVE} to Dropbox"
echo "2. Get shareable download link from Dropbox"
echo "3. Update DROPBOX_URL in scripts/download_data.sh"
echo ""
echo "Note: Make sure to get a DIRECT download link, not a preview link."
echo "Dropbox links should end with '?dl=1' for direct download."
echo "==========================================="
