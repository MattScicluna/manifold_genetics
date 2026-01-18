#!/bin/bash
#
# External tools setup script for manifold-genetics
#
# This script downloads external command-line tools (plink2, flashPCA).
# It does NOT manage the Python environment.
#
# For Python environment setup, see README.md and use:
#   uv venv --python python3.11
#   uv sync --frozen
#
# Usage:
#   bash setup.sh
#
# IMPORTANT: Run this on LOGIN NODE (has internet access)
#

set -e  # Exit on error

echo "=========================================="
echo "manifold-genetics: External Tools Setup"
echo "=========================================="
echo ""
echo "NOTE: This script does NOT manage the Python environment."
echo "Use 'uv sync --frozen' as described in the README."
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: Must run from manifold_genetics directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "=========================================="
echo "Downloading External Tools"
echo "=========================================="
echo ""
echo "IMPORTANT: Compute nodes have no internet access."
echo "Downloading plink2 and flashPCA now..."
echo ""

mkdir -p bin

# Download plink2
if [ ! -f "bin/plink2" ]; then
    echo "Downloading plink2..."
    PLINK_URL="https://s3.amazonaws.com/plink2-assets/plink2_linux_x86_64_20260110.zip"

    # Try wget first (better for binary downloads), then curl
    if command -v wget &> /dev/null; then
        wget -O bin/plink2.zip "$PLINK_URL"
    else
        curl -L -o bin/plink2.zip "$PLINK_URL"
    fi

    # Verify it's actually a zip file
    if file bin/plink2.zip | grep -q "Zip archive"; then
        unzip -o bin/plink2.zip -d bin/
        chmod +x bin/plink2
        rm bin/plink2.zip
        echo "✓ plink2 downloaded: $(ls -lh bin/plink2 | awk '{print $5}')"
    else
        echo "ERROR: Downloaded file is not a valid zip archive"
        echo "File type: $(file bin/plink2.zip)"
        echo ""
        echo "Trying direct binary download instead..."
        # Try alternative: download pre-extracted binary
        PLINK_BIN_URL="https://s3.amazonaws.com/plink2-assets/plink2_linux_x86_64_latest.zip"
        rm -f bin/plink2.zip
        wget -O bin/plink2.zip "$PLINK_BIN_URL" || curl -L -o bin/plink2.zip "$PLINK_BIN_URL"

        if file bin/plink2.zip | grep -q "Zip archive"; then
            unzip -o bin/plink2.zip -d bin/
            chmod +x bin/plink2
            rm bin/plink2.zip
            echo "✓ plink2 downloaded (alternative URL)"
        else
            echo "ERROR: Could not download plink2. Please download manually from:"
            echo "  https://www.cog-genomics.org/plink/2.0/"
            echo "  and place the binary in: $(pwd)/bin/plink2"
            exit 1
        fi
    fi
else
    echo "✓ plink2 already exists"
fi

# Download flashPCA
if [ ! -f "bin/flashpca" ] && [ ! -f "bin/flashpca_x86-64" ]; then
    echo "Downloading flashPCA..."
    FLASHPCA_URL="https://github.com/gabraham/flashpca/releases/download/v2.0/flashpca_x86-64.gz"

    # Try wget first, then curl
    if command -v wget &> /dev/null; then
        wget -O bin/flashpca_x86-64.gz "$FLASHPCA_URL"
    else
        curl -L -o bin/flashpca_x86-64.gz "$FLASHPCA_URL"
    fi

    gunzip bin/flashpca_x86-64.gz
    chmod +x bin/flashpca_x86-64
    ln -sf flashpca_x86-64 bin/flashpca
    echo "✓ flashPCA downloaded: $(ls -lh bin/flashpca_x86-64 | awk '{print $5}')"
else
    echo "✓ flashPCA already exists"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Downloaded tools to bin/:"
ls -lh bin/ 2>/dev/null || echo "  (no files yet)"
echo ""
echo "Total size: $(du -sh bin/ 2>/dev/null | awk '{print $1}' || echo '0')"
echo ""
echo "Next steps:"
echo ""
echo "1. Set up Python environment (if not already done):"
echo "   uv venv --python python3.11"
echo "   uv sync --frozen"
echo ""
echo "2. Run tests:"
echo "   uv run pytest -m 'not slow and not network'"
echo ""
echo "3. Download and process example data:"
echo "   bash examples/hgdp_1kgp/download_data.sh"
echo "   bash examples/hgdp_1kgp/prepare_data.sh"
echo ""
echo "4. Run example pipeline:"
echo "   bash examples/hgdp_1kgp/run_pipeline.sh"
echo ""
echo "All tools are pre-downloaded and ready to use!"
echo "=========================================="
