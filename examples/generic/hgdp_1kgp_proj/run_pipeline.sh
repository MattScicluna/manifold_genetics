#!/bin/bash
#
# Generic Cross-Projection Pipeline Script
#
# This script runs the complete cross-projection analysis pipeline:
# - PCA with fit/project workflow
# - Neural Admixture
# - Embedding (PHATE, UMAP, etc.)
# - Visualization
#
# Usage:
#   bash run_pipeline.sh \
#       --fit-plink ./data/fit_subset \
#       --project-plink ./data/project_subset \
#       --fit-labels ./data/fit_labels.csv \
#       --project-labels ./data/project_labels.csv \
#       --fit-colormap /path/to/fit_colormap.json \
#       --project-colormap /path/to/project_colormap.json \
#       --output-dir ./outputs
#

set -e

# ============================================================================
# Argument Parsing
# ============================================================================

# Default values
FIT_PLINK=""
PROJECT_PLINK=""
FIT_LABELS=""
PROJECT_LABELS=""
FIT_COLORMAP=""
PROJECT_COLORMAP=""
OUTPUT_DIR="./outputs"
N_PCS=20
K_MIN=2
K_MAX=10
EMBEDDING_METHOD="phate"
KNN=100
T=3
ADMIXTURE_GROUP_COLUMN=""
THREADS=4
NEURALADMIXTURE_BATCH_SIZE=400
EMBED_BATCH_SIZE=""
SKIP_METRICS=false
NUM_GPUS=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fit-plink)
            FIT_PLINK="$2"
            shift 2
            ;;
        --project-plink)
            PROJECT_PLINK="$2"
            shift 2
            ;;
        --fit-labels)
            FIT_LABELS="$2"
            shift 2
            ;;
        --project-labels)
            PROJECT_LABELS="$2"
            shift 2
            ;;
        --fit-colormap)
            FIT_COLORMAP="$2"
            shift 2
            ;;
        --project-colormap)
            PROJECT_COLORMAP="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --n-pcs)
            N_PCS="$2"
            shift 2
            ;;
        --k-min)
            K_MIN="$2"
            shift 2
            ;;
        --k-max)
            K_MAX="$2"
            shift 2
            ;;
        --embedding)
            EMBEDDING_METHOD="$2"
            shift 2
            ;;
        --knn)
            KNN="$2"
            shift 2
            ;;
        --t)
            T="$2"
            shift 2
            ;;
        --admixture-group-column)
            ADMIXTURE_GROUP_COLUMN="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --neuraladmixture-batch-size)
            NEURALADMIXTURE_BATCH_SIZE="$2"
            shift 2
            ;;
        --embed-batch-size)
            EMBED_BATCH_SIZE="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --skip-metrics)
            SKIP_METRICS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --fit-plink PATH --project-plink PATH --fit-labels PATH --project-labels PATH --fit-colormap PATH --project-colormap PATH --output-dir PATH [OPTIONS]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$FIT_PLINK" ]] || [[ -z "$PROJECT_PLINK" ]] || [[ -z "$FIT_LABELS" ]] || [[ -z "$PROJECT_LABELS" ]] || [[ -z "$FIT_COLORMAP" ]] || [[ -z "$PROJECT_COLORMAP" ]]; then
    echo "Error: Missing required arguments"
    echo ""
    echo "Usage: $0 \\"
    echo "    --fit-plink PATH                     # Fit PLINK prefix"
    echo "    --project-plink PATH                 # Project PLINK prefix"
    echo "    --fit-labels PATH                    # Fit labels CSV"
    echo "    --project-labels PATH                # Project labels CSV"
    echo "    --fit-colormap PATH                  # Fit colormap JSON"
    echo "    --project-colormap PATH              # Project colormap JSON"
    echo "    --output-dir PATH                    # Output directory"
    echo "    [--n-pcs N]                          # Number of PCA components (default: 20)"
    echo "    [--k-min K]                          # Min admixture K (default: 2)"
    echo "    [--k-max K]                          # Max admixture K (default: 10)"
    echo "    [--embedding METHOD]                 # Embedding method (default: phate)"
    echo "    [--knn N]                            # KNN parameter (default: 100)"
    echo "    [--t N]                              # Diffusion time (default: 3)"
    echo "    [--admixture-group-column COL]       # Metadata column for admixture"
    echo "    [--threads N]                        # Threads (default: 4)"
    echo "    [--neuraladmixture-batch-size N]     # Batch size (default: 400)"
    echo "    [--embed-batch-size N]               # Embedding batch size"
    echo "    [--num-gpus N]                       # Number of GPUs"
    echo "    [--skip-metrics]                     # Skip metrics calculation"
    exit 1
fi

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ANSI colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Print header
echo ""
echo "================================================="
echo "  Generic Cross-Projection Pipeline"
echo "================================================="
echo ""

# ============================================================================
# Step 1: Verify Required Files
# ============================================================================
print_status "Verifying required files..."

REQUIRED_FILES=(
    "${FIT_PLINK}.bed"
    "${FIT_PLINK}.bim"
    "${FIT_PLINK}.fam"
    "${PROJECT_PLINK}.bed"
    "${PROJECT_PLINK}.bim"
    "${PROJECT_PLINK}.fam"
    "${FIT_LABELS}"
    "${PROJECT_LABELS}"
    "${FIT_COLORMAP}"
    "${PROJECT_COLORMAP}"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        MISSING_FILES+=("$file")
    fi
done

if [[ ${#MISSING_FILES[@]} -gt 0 ]]; then
    echo "Missing required files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Please run prepare_data.sh first"
    exit 1
fi

print_success "All required files found"

# ============================================================================
# Step 2: Check Virtual Environment
# ============================================================================
print_status "Checking virtual environment..."

if [[ -z "$VIRTUAL_ENV" ]]; then
    print_warning "Virtual environment not activated"
    echo ""
    echo "Please activate the virtual environment first:"
    echo "  cd ${PROJECT_ROOT}"
    echo "  source .venv/bin/activate"
    echo ""
    exit 1
fi

print_success "Virtual environment active: $VIRTUAL_ENV"

# ============================================================================
# Step 3: Get Data Statistics
# ============================================================================
print_status "Getting data statistics..."

FIT_SAMPLES=$(wc -l < "${FIT_PLINK}.fam")
PROJECT_SAMPLES=$(wc -l < "${PROJECT_PLINK}.fam")
SNP_COUNT=$(wc -l < "${FIT_PLINK}.bim")

echo "  Fit dataset: $FIT_SAMPLES samples"
echo "  Project dataset: $PROJECT_SAMPLES samples"
echo "  Common SNPs: $SNP_COUNT SNPs"

# ============================================================================
# Step 4: Run Pipeline
# ============================================================================
print_status "Running cross-projection analysis pipeline..."

echo ""
echo "Command:"
echo "manifold-genetics pipeline \\"
echo "    --fit-plink ${FIT_PLINK} \\"
echo "    --project-plink ${PROJECT_PLINK} \\"
echo "    --fit-labels ${FIT_LABELS} \\"
echo "    --project-labels ${PROJECT_LABELS} \\"
echo "    --fit-colormap ${FIT_COLORMAP} \\"
echo "    --project-colormap ${PROJECT_COLORMAP} \\"
echo "    --output ${OUTPUT_DIR} \\"
echo "    --n-pcs ${N_PCS} \\"
echo "    --k-min ${K_MIN} --k-max ${K_MAX} \\"
echo "    --embedding ${EMBEDDING_METHOD} --knn ${KNN} --t ${T} \\"
[[ -n "$ADMIXTURE_GROUP_COLUMN" ]] && echo "    --admixture-group-column ${ADMIXTURE_GROUP_COLUMN} \\"
echo "    --threads ${THREADS} \\"
echo "    --neuraladmixture-batch-size ${NEURALADMIXTURE_BATCH_SIZE}"
[[ -n "$EMBED_BATCH_SIZE" ]] && echo "    --embed-batch-size ${EMBED_BATCH_SIZE}"
[[ -n "$NUM_GPUS" ]] && echo "    --num-gpus ${NUM_GPUS}"
[[ "$SKIP_METRICS" == "true" ]] && echo "    --skip-metrics"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="manifold-genetics pipeline"
CMD="$CMD --fit-plink \"$FIT_PLINK\""
CMD="$CMD --project-plink \"$PROJECT_PLINK\""
CMD="$CMD --fit-labels \"$FIT_LABELS\""
CMD="$CMD --project-labels \"$PROJECT_LABELS\""
CMD="$CMD --fit-colormap \"$FIT_COLORMAP\""
CMD="$CMD --project-colormap \"$PROJECT_COLORMAP\""
CMD="$CMD --output \"$OUTPUT_DIR\""
CMD="$CMD --n-pcs $N_PCS"
CMD="$CMD --k-min $K_MIN --k-max $K_MAX"
CMD="$CMD --embedding $EMBEDDING_METHOD --knn $KNN --t $T"
[[ -n "$ADMIXTURE_GROUP_COLUMN" ]] && CMD="$CMD --admixture-group-column $ADMIXTURE_GROUP_COLUMN"
CMD="$CMD --threads $THREADS"
CMD="$CMD --neuraladmixture-batch-size $NEURALADMIXTURE_BATCH_SIZE"
[[ -n "$EMBED_BATCH_SIZE" ]] && CMD="$CMD --embed-batch-size $EMBED_BATCH_SIZE"
[[ -n "$NUM_GPUS" ]] && CMD="$CMD --num-gpus $NUM_GPUS"
[[ "$SKIP_METRICS" == "true" ]] && CMD="$CMD --skip-metrics"

# Run pipeline
eval $CMD

# ============================================================================
# Step 5: Summary
# ============================================================================
echo ""
echo "================================================="
print_success "Cross-projection pipeline analysis complete!"
echo "================================================="
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
echo "  📊 PCA (fit → project):"
echo "    - pca/fit_pca_${N_PCS}.csv, pca/transform_pca_${N_PCS}.csv"
echo "    - figures/pca/"
echo ""
echo "  🧬 Admixture (K=${K_MIN}-${K_MAX}):"
echo "    - admixture/fit.{${K_MIN}..${K_MAX}}.csv, admixture/transform.{${K_MIN}..${K_MAX}}.csv"
echo "    - figures/admixture/"
echo ""
echo "  🗺️ ${EMBEDDING_METHOD^^} Embedding:"
echo "    - embeddings/${EMBEDDING_METHOD}_2d.csv"
echo "    - figures/embeddings/"
echo ""
if [[ "$SKIP_METRICS" != "true" ]]; then
    echo "  📈 Metrics:"
    echo "    - metrics/geographic.json (if geographic data available)"
    echo "    - metrics/admixture.json"
    echo ""
fi
echo ""
