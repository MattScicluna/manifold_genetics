#!/bin/bash
#
# Quick setup script for manifold-genetics
#
# Usage:
#   bash setup.sh
#
# IMPORTANT: Run this on LOGIN NODE (has internet access)
#

set -e  # Exit on error

echo "=========================================="
echo "manifold-genetics: Setup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: Must run from manifold_genetics directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Select Python interpreter (prefer 3.11 for torch/neural-admixture wheels)
PYTHON_BIN=""
if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
elif command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    if command -v uv &> /dev/null; then
        if [ -n "$PYTHON_BIN" ]; then
            uv venv --python "$PYTHON_BIN"
        else
            echo "WARNING: python3.11/3.10 not found; using default python for venv."
            uv venv
        fi
    else
        if [ -n "$PYTHON_BIN" ]; then
            "$PYTHON_BIN" -m venv .venv
        else
            echo "ERROR: python3.11 or python3.10 not found. Please load a module or install one, then re-run setup."
            exit 1
        fi
    fi
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install package
echo "Installing manifold-genetics..."
if command -v uv &> /dev/null; then
    uv pip install -e .
else
    pip install -e .
fi

# Download external tools (CRITICAL: compute nodes have no internet!)
echo ""
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

# Set environment variables for tools
export PLINK_PATH="$(pwd)/bin/plink2"
export FLASHPCA_PATH="$(pwd)/bin/flashpca_x86-64"

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
echo "1. Request compute node:"
echo "   salloc --account=ctb-hussinju --cpus-per-task=4 --mem=16GB --time=1:00:00"
echo ""
echo "2. On compute node, activate environment:"
echo "   cd $(pwd)"
echo "   source .venv/bin/activate"
echo ""
echo "3. Run tests:"
echo "   pytest"
echo ""
echo "4. Download and process example data:"
echo "   bash examples/hgdp_1kgp/download_data.sh"
echo "   bash examples/hgdp_1kgp/prepare_data.sh"
echo ""
echo "5. Run example pipeline:"
echo "   bash examples/hgdp_1kgp/run_pipeline.sh"
echo ""
echo "All tools are pre-downloaded and ready to use!"
echo "=========================================="
