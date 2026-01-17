#!/bin/bash
#
# Download HGDP+1KGP data from Dropbox
#
# This script downloads the full dataset archive from Dropbox and extracts it.
# Run on LOGIN NODE (requires internet access).
#
# Smart caching: Skips download if data already exists.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==========================================="
echo "Download HGDP+1KGP Data"
echo "==========================================="
echo ""

# Configuration
# Note: Dropbox URLs need "?dl=1" at the end to direct download (not "?dl=0")
DROPBOX_URL="https://www.dropbox.com/scl/fi/f8kay7kgpgb2s5ncsykpg/hgdp_1kgp_full.tar.gz?rlkey=pp7fj711zxlg4o2lbskxj5rwe&st=jv1xh8vo&dl=1"

DATA_DIR="${SCRIPT_DIR}/data"
RAW_DIR="${DATA_DIR}/raw"
ARCHIVE_NAME="hgdp_1kgp_full.tar.gz"
ARCHIVE_PATH="${DATA_DIR}/${ARCHIVE_NAME}"

echo "Step 1: Check if data already exists"
echo "--------------------------------------"

# Check if processed data already exists (highest priority check)
if [ -f "${DATA_DIR}/fit_subset.bed" ] && [ -f "${DATA_DIR}/transform_subset.bed" ]; then
    echo "✓ Data already processed"
    echo ""
    echo "Found:"
    echo "  - ${DATA_DIR}/fit_subset.bed"
    echo "  - ${DATA_DIR}/transform_subset.bed"
    echo ""
    echo "Nothing to download. Ready to run tests!"
    exit 0
fi

# Check if raw data already exists
if [ -f "${RAW_DIR}/full_dataset.bed" ] && \
   [ -f "${RAW_DIR}/full_dataset.bim" ] && \
   [ -f "${RAW_DIR}/full_dataset.fam" ]; then
    echo "✓ Raw data already downloaded"
    echo ""
    echo "Found:"
    echo "  - ${RAW_DIR}/full_dataset.bed"
    echo "  - ${RAW_DIR}/full_dataset.bim"
    echo "  - ${RAW_DIR}/full_dataset.fam"
    echo ""
    echo "Raw data ready. Next step:"
    echo "  bash examples/hgdp_1kgp/prepare_data.sh"
    exit 0
fi

# Check if archive exists but hasn't been extracted
if [ -f "$ARCHIVE_PATH" ]; then
    echo "✓ Archive already downloaded: ${ARCHIVE_NAME}"
    echo "  Skipping download, will extract..."
    SKIP_DOWNLOAD=true
else
    SKIP_DOWNLOAD=false
fi

echo ""
echo "Step 2: Validate configuration"
echo "--------------------------------------"

if [ "$DROPBOX_URL" = "PLACEHOLDER_URL" ]; then
    echo "ERROR: Dropbox URL not configured!"
    echo ""
    echo "Please update the DROPBOX_URL variable in this script with the actual download link."
    echo ""
    echo "Edit: $0"
    echo "Line: DROPBOX_URL=\"PLACEHOLDER_URL\""
    echo ""
    exit 1
fi

echo "✓ Download URL configured"

echo ""
echo "Step 3: Create directories"
echo "--------------------------------------"

mkdir -p "$DATA_DIR"
mkdir -p "$RAW_DIR"
echo "✓ Created: ${DATA_DIR}"
echo "✓ Created: ${RAW_DIR}"

if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo "Step 4: Download archive from Dropbox"
    echo "--------------------------------------"

    echo "Downloading: ${ARCHIVE_NAME}"
    echo "Destination: ${ARCHIVE_PATH}"
    echo "Size: ~183 MB (may take a few minutes)"
    echo ""

    # Try wget first (better progress bar), then curl
    if command -v wget &> /dev/null; then
        wget -O "$ARCHIVE_PATH" "$DROPBOX_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$ARCHIVE_PATH" "$DROPBOX_URL"
    else
        echo "ERROR: Neither wget nor curl found!"
        echo "Please install wget or curl to download files."
        exit 1
    fi

    echo ""
    echo "✓ Download complete"

    echo ""
    echo "Step 5: Verify download"
    echo "--------------------------------------"

    # Check file exists and has reasonable size (> 10 MB, since expected ~183 MB)
    if [ ! -f "$ARCHIVE_PATH" ]; then
        echo "ERROR: Downloaded file not found: $ARCHIVE_PATH"
        exit 1
    fi

    FILE_SIZE=$(stat -f%z "$ARCHIVE_PATH" 2>/dev/null || stat -c%s "$ARCHIVE_PATH" 2>/dev/null || echo "0")
    FILE_SIZE_MB=$((FILE_SIZE / 1024 / 1024))

    if [ "$FILE_SIZE_MB" -lt 10 ]; then
        echo "ERROR: Downloaded file is too small (${FILE_SIZE_MB} MB)"
        echo "Expected ~183 MB. Download may have failed."
        echo "Check the Dropbox URL and try again."
        exit 1
    fi

    echo "✓ File size: ${FILE_SIZE_MB} MB"
else
    echo ""
    echo "Step 4: Download (skipped)"
    echo "--------------------------------------"
    echo "✓ Using existing archive"

    echo ""
    echo "Step 5: Verify archive (skipped)"
    echo "--------------------------------------"
fi

echo ""
echo "Step 6: Extract archive"
echo "--------------------------------------"

echo "Extracting to: ${RAW_DIR}"
tar -xzf "$ARCHIVE_PATH" -C "$RAW_DIR"
echo "✓ Extraction complete"

echo ""
echo "Step 7: Verify extracted files"
echo "--------------------------------------"

# Check all required files
REQUIRED_FILES=(
    "full_dataset.bed"
    "full_dataset.bim"
    "full_dataset.fam"
    "metadata.csv"
    "README.txt"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "${RAW_DIR}/${file}" ]; then
        echo "✓ ${file}"
    else
        echo "✗ ${file} (MISSING)"
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo ""
    echo "ERROR: Some files are missing from the archive!"
    echo "Missing: ${MISSING_FILES[*]}"
    echo ""
    echo "The archive may be corrupted. Try:"
    echo "1. Delete the archive: rm ${ARCHIVE_PATH}"
    echo "2. Run this script again to re-download"
    exit 1
fi

echo ""
echo "Step 8: Show file sizes"
echo "--------------------------------------"

du -h "${RAW_DIR}/"* | awk '{printf "  %-15s %s\n", $1, $2}'

echo ""
echo "==========================================="
echo "Download Complete!"
echo "==========================================="
echo ""
echo "Raw data location:"
echo "  ${RAW_DIR}/"
echo ""
echo "Files downloaded:"
echo "  - full_dataset.{bed,bim,fam} (4,151 samples, 172,152 SNPs)"
echo "  - metadata.csv (sample information and QC flags)"
echo "  - README.txt (data description)"
echo ""
echo "Next step:"
echo "  bash examples/hgdp_1kgp/prepare_data.sh"
echo ""
echo "This will create analysis-ready subsets:"
echo "  - fit_subset (3,400 unrelated samples)"
echo "  - transform_subset (4,094 QC-passing samples)"
echo "==========================================="
