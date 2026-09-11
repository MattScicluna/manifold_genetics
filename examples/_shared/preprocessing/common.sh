#!/bin/bash
#
# Shared Preprocessing Utilities
#
# This file contains common utility functions used by preprocessing scripts.
# Source this file in your scripts:
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#
# Provides:
#   - ANSI color printing functions
#   - Tool discovery (plink, plink2)
#   - PLINK file verification
#   - Statistics helpers
#

# ============================================================================
# ANSI Colors
# ============================================================================
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ============================================================================
# Printing Functions
# ============================================================================

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

print_header() {
    echo ""
    echo "=========================================="
    echo "  $1"
    echo "=========================================="
    echo ""
}

print_subheader() {
    echo ""
    echo "----------------------------------------"
    echo "  $1"
    echo "----------------------------------------"
}

# ============================================================================
# Tool Discovery
# ============================================================================

# Find plink2 binary
# Sets global variable PLINK2 on success
# Args:
#   $1 - PROJECT_ROOT (optional, defaults to detecting from script location)
# Returns:
#   0 on success, 1 on failure
find_plink2() {
    local project_root="${1:-}"

    # Try to detect project root if not provided
    if [[ -z "$project_root" ]]; then
        # Look for bin/plink2 relative to script
        local script_dir
        script_dir="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
        # Walk up to find project root (contains bin/ or src/)
        local dir="$script_dir"
        while [[ "$dir" != "/" ]]; do
            if [[ -d "$dir/bin" ]] || [[ -d "$dir/src" ]]; then
                project_root="$dir"
                break
            fi
            dir="$(dirname "$dir")"
        done
    fi

    PLINK2=""

    # Check bin/plink2
    if [[ -n "$project_root" ]] && [[ -f "${project_root}/bin/plink2" ]]; then
        PLINK2="${project_root}/bin/plink2"
        print_success "Using plink2 from bin/plink2"
    # Check loaded modules
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
        echo "  1. Run 'uv run manifold-genetics setup' to download to bin/plink2 (recommended)"
        echo "  2. Load plink2 module: module load plink2"
        echo "  3. Add plink2 to your PATH"
        echo ""
        return 1
    fi

    # Verify it's executable
    if ! ${PLINK2} --version &> /dev/null; then
        print_error "plink2 found but not executable!"
        return 1
    fi

    return 0
}

# Find plink (v1.9) binary
# Sets global variable PLINK on success
# Args:
#   $1 - PROJECT_ROOT (optional)
# Returns:
#   0 on success, 1 on failure
find_plink() {
    local project_root="${1:-}"

    # Try to detect project root if not provided
    if [[ -z "$project_root" ]]; then
        local script_dir
        script_dir="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
        local dir="$script_dir"
        while [[ "$dir" != "/" ]]; do
            if [[ -d "$dir/bin" ]] || [[ -d "$dir/src" ]]; then
                project_root="$dir"
                break
            fi
            dir="$(dirname "$dir")"
        done
    fi

    PLINK=""

    # Check bin/plink
    if [[ -n "$project_root" ]] && [[ -f "${project_root}/bin/plink" ]]; then
        PLINK="${project_root}/bin/plink"
        print_success "Using plink from bin/plink"
    # Check loaded modules
    elif module list 2>&1 | grep -q "plink/"; then
        PLINK="plink"
        print_success "Using plink from loaded module"
    # Check PATH
    elif command -v plink &> /dev/null; then
        PLINK="plink"
        print_success "Using plink from PATH"
    else
        print_error "plink (v1.9) not found!"
        echo ""
        echo "Please ensure plink is available via one of:"
        echo "  1. Run 'uv run manifold-genetics setup' to download to bin/plink (recommended)"
        echo "  2. Load plink module: module load plink/1.9"
        echo "  3. Add plink to your PATH"
        echo ""
        return 1
    fi

    # Verify it's executable
    if ! ${PLINK} --version &> /dev/null; then
        print_error "plink found but not executable!"
        return 1
    fi

    return 0
}

# ============================================================================
# PLINK File Verification
# ============================================================================

# Verify PLINK triplet exists (.bed, .bim, .fam)
# Args:
#   $1 - PLINK prefix (without extension)
# Returns:
#   0 if all files exist, 1 otherwise
verify_plink_files() {
    local prefix="$1"
    local missing=()

    for ext in bed bim fam; do
        if [[ ! -f "${prefix}.${ext}" ]]; then
            missing+=("${prefix}.${ext}")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        print_error "Missing PLINK files:"
        for file in "${missing[@]}"; do
            echo "  - $file"
        done
        return 1
    fi

    return 0
}

# ============================================================================
# Statistics Helpers
# ============================================================================

# Get sample count from .fam file
# Args:
#   $1 - PLINK prefix
get_sample_count() {
    local prefix="$1"
    wc -l < "${prefix}.fam"
}

# Get SNP count from .bim file
# Args:
#   $1 - PLINK prefix
get_snp_count() {
    local prefix="$1"
    wc -l < "${prefix}.bim"
}

# Print PLINK dataset statistics
# Args:
#   $1 - PLINK prefix
#   $2 - Dataset name (optional, for display)
print_plink_stats() {
    local prefix="$1"
    local name="${2:-Dataset}"

    local samples snps
    samples=$(get_sample_count "$prefix")
    snps=$(get_snp_count "$prefix")

    echo "  ${name}: ${samples} samples, ${snps} SNPs"
}

# ============================================================================
# Path Helpers
# ============================================================================

# Get the project root directory
# Returns path via stdout
get_project_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
    local dir="$script_dir"
    while [[ "$dir" != "/" ]]; do
        if [[ -d "$dir/src/manifold_genetics" ]]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    # Fallback
    echo "$script_dir"
}

# Get the shared directory
# Returns path via stdout
get_shared_dir() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
    local dir="$script_dir"
    while [[ "$dir" != "/" ]]; do
        if [[ -d "$dir/_shared" ]]; then
            echo "$dir/_shared"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    # Fallback - assume we're in examples/
    echo "$(dirname "$script_dir")/_shared"
}

# ---------------------------------------------------------------------------
# Label freshness
# ---------------------------------------------------------------------------

# labels_match_fam <labels.csv> <subset.fam>
#
# Exit 0 when the label file covers the samples currently in the .fam.
#
# Every prepare_data.sh used to skip label generation whenever the file merely
# existed. That is how examples/ukbb/geosketch_phate went wrong: re-running the
# selection produced a new .fam on 2026-05-16 while the 2026-05-12 label file was
# left in place. They overlapped by 40.6%, and the pipeline went on to publish
# figures that coloured 40% of their points. Existence is not freshness.
#
# Deliberately stdlib-only. A freshness check must not depend on pandas being
# importable by whichever python3 is first on PATH -- and since any failure here
# exits non-zero, i.e. "regenerate", an ImportError would otherwise masquerade
# as staleness and silently force a rebuild every time.
labels_match_fam() {
    local label_file="$1"
    local fam_file="$2"

    [[ -f "$label_file" && -f "$fam_file" ]] || return 1

    python3 - "$label_file" "$fam_file" <<'MG_FRESHNESS_CHECK'
import csv
import os
import sys

labels_path, fam_path = sys.argv[1], sys.argv[2]

try:
    with open(labels_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            print("    no sample_id column; regenerating")
            sys.exit(1)
        have = {row["sample_id"] for row in reader}

    wanted = set()
    with open(fam_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                wanted.add(parts[1])
except OSError as exc:
    print(f"    could not read labels or .fam ({exc}); regenerating")
    sys.exit(1)

missing = wanted - have
if missing:
    pct = 100.0 * (len(wanted) - len(missing)) / len(wanted) if wanted else 0.0
    print(
        f"    stale: {len(missing)} of {len(wanted)} samples in "
        f"{os.path.basename(fam_path)} have no label ({pct:.1f}% covered)"
    )
    sys.exit(1)
sys.exit(0)
MG_FRESHNESS_CHECK
}
