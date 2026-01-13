#!/bin/bash
#
# Shared Pipeline Runner
#
# This script provides a unified interface for running the manifold-genetics pipeline
# across all datasets. It supports three pipeline modes:
#
#   1. projection: Cross-cohort projection (fit on one cohort, project on another)
#   2. subsample: Within-cohort subsampling (fit on subsample, project on full dataset)
#   3. transform: Fit and transform on target dataset only (no cross-projection)
#
# Usage:
#   run_pipeline.sh --mode [projection|subsample|transform] [OPTIONS]
#
# DESIGN PRINCIPLE:
#   - Mode controls SEMANTIC BEHAVIOR (what to fit/transform on)
#   - Scripts override PERFORMANCE PARAMETERS (knn/t/landmarking) based on dataset size
#
# Mode-specific defaults:
#   projection: --embedding-input both --knn 100 --t 3 (no landmarking)
#               Semantic: fit embedding on reference, transform on target
#               Use Case: Projecting a large biobank (UKBB, AoU) onto a smaller reference dataset (HGDP+1KGP)
#
#   subsample:  --embedding-input fit --knn 500 --t 50 --n-landmark 10000 --random-landmarking
#               Semantic: fit+transform on subsampled fit set only (cheaper)
#               Use Case: visualize large biobanks (UKBB, AoU) without any reference set.
#
#   transform:  --embedding-input project --knn 100 --t 3 (no landmarking)
#               Semantic: fit+transform embedding on target dataset only
#               Use case: Use when transform set contains fit set, so no need to re-project (HGDP+1KGP)
#
# All mode defaults can be overridden with explicit arguments.
#

set -e

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

# Mode-specific defaults
set_projection_mode_defaults() {
    # Cross-projection: fit embedding on reference (HGDP+1KGP), transform on cohort (UKBB/AoU)
    EMBEDDING_INPUT="${EMBEDDING_INPUT:-both}"
    KNN="${KNN:-100}"
    T="${T:-3}"
    N_LANDMARK="${N_LANDMARK:-}"
    RANDOM_LANDMARKING="${RANDOM_LANDMARKING:-false}"
    NEURALADMIXTURE_BATCH_SIZE="${NEURALADMIXTURE_BATCH_SIZE:-}"
    EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-}"
}

set_subsample_mode_defaults() {
    # Within-cohort: default is fit on subsample only (cheaper)
    # Override with --embedding-input both to project on full cohort (more expensive)
    EMBEDDING_INPUT="${EMBEDDING_INPUT:-fit}"
    KNN="${KNN:-500}"
    T="${T:-50}"
    N_LANDMARK="${N_LANDMARK:-10000}"
    RANDOM_LANDMARKING="${RANDOM_LANDMARKING:-true}"
    NEURALADMIXTURE_BATCH_SIZE="${NEURALADMIXTURE_BATCH_SIZE:-400}"
    EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-}"
}

set_transform_mode_defaults() {
    # Fit and transform on target dataset only: when transform set contains fit set
    # No cross-projection needed - just fit+transform on the same target dataset
    EMBEDDING_INPUT="${EMBEDDING_INPUT:-project}"
    KNN="${KNN:-100}"
    T="${T:-3}"
    N_LANDMARK="${N_LANDMARK:-}"
    RANDOM_LANDMARKING="${RANDOM_LANDMARKING:-false}"
    NEURALADMIXTURE_BATCH_SIZE="${NEURALADMIXTURE_BATCH_SIZE:-}"
    EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-}"
}

# Parse arguments
MODE=""
FIT_PLINK=""
PROJECT_PLINK=""
LABELS=""
FIT_LABELS=""
PROJECT_LABELS=""
COLORMAP=""
FIT_COLORMAP=""
PROJECT_COLORMAP=""
OUTPUT=""
GEOGRAPHIC=""
N_PCS=50
K_MIN=2
K_MAX=10
EMBEDDING="phate"
EMBEDDING_INPUT=""
KNN=""
T=""
N_LANDMARK=""
RANDOM_LANDMARKING=""
ADMIXTURE_GROUP_COLUMN=""
ADMIXTURE_WITHIN_GROUP_ORDER="chron"
THREADS=""
NUM_GPUS=""
NEURALADMIXTURE_BATCH_SIZE=""
EMBED_BATCH_SIZE=""
FLASHPCA_OUTPUT_DIR=""
SKIP_METRICS=false
SKIP_PCA=false
SKIP_ADMIXTURE=false
SKIP_EMBEDDING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --fit-plink)
            FIT_PLINK="$2"
            shift 2
            ;;
        --project-plink)
            PROJECT_PLINK="$2"
            shift 2
            ;;
        --labels)
            LABELS="$2"
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
        --colormap)
            COLORMAP="$2"
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
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --geographic)
            GEOGRAPHIC="$2"
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
            EMBEDDING="$2"
            shift 2
            ;;
        --embedding-input)
            EMBEDDING_INPUT="$2"
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
        --n-landmark)
            N_LANDMARK="$2"
            shift 2
            ;;
        --random-landmarking)
            RANDOM_LANDMARKING="true"
            shift
            ;;
        --admixture-group-column)
            ADMIXTURE_GROUP_COLUMN="$2"
            shift 2
            ;;
        --admixture-within-group-order)
            ADMIXTURE_WITHIN_GROUP_ORDER="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
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
        --flashpca-output-dir)
            FLASHPCA_OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-metrics)
            SKIP_METRICS=true
            shift
            ;;
        --skip-pca)
            SKIP_PCA=true
            shift
            ;;
        --skip-admixture)
            SKIP_ADMIXTURE=true
            shift
            ;;
        --skip-embedding)
            SKIP_EMBEDDING=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Usage: $0 --mode [projection|subsample|transform] --fit-plink PATH --project-plink PATH [OPTIONS]"
            exit 1
            ;;
    esac
done

# Apply mode-specific defaults
if [[ "$MODE" == "projection" ]]; then
    set_projection_mode_defaults
elif [[ "$MODE" == "subsample" ]]; then
    set_subsample_mode_defaults
elif [[ "$MODE" == "transform" ]]; then
    set_transform_mode_defaults
elif [[ -z "$MODE" ]]; then
    print_warning "No --mode specified. Using explicit parameters or falling back to projection mode defaults."
    set_projection_mode_defaults
else
    print_error "Invalid mode: $MODE (must be 'projection', 'subsample', or 'transform')"
    exit 1
fi

# Handle labels/colormap: prefer shared, fall back to separate
if [[ -n "$LABELS" ]]; then
    FIT_LABELS="${FIT_LABELS:-$LABELS}"
    PROJECT_LABELS="${PROJECT_LABELS:-$LABELS}"
fi

if [[ -n "$COLORMAP" ]]; then
    FIT_COLORMAP="${FIT_COLORMAP:-$COLORMAP}"
    PROJECT_COLORMAP="${PROJECT_COLORMAP:-$COLORMAP}"
fi

# Validate required arguments
if [[ -z "$FIT_PLINK" ]] || [[ -z "$PROJECT_PLINK" ]] || [[ -z "$OUTPUT" ]]; then
    print_error "Missing required arguments"
    echo ""
    echo "Usage: $0 \\"
    echo "    --mode [projection|subsample|transform] \\"
    echo "    --fit-plink PATH \\"
    echo "    --project-plink PATH \\"
    echo "    --output PATH \\"
    echo "    --labels PATH (or --fit-labels + --project-labels) \\"
    echo "    --colormap PATH (or --fit-colormap + --project-colormap) \\"
    echo "    [OPTIONS]"
    exit 1
fi

if [[ -z "$FIT_LABELS" ]] || [[ -z "$PROJECT_LABELS" ]]; then
    print_error "Missing labels: provide --labels OR both --fit-labels and --project-labels"
    exit 1
fi

if [[ -z "$FIT_COLORMAP" ]] || [[ -z "$PROJECT_COLORMAP" ]]; then
    print_error "Missing colormap: provide --colormap OR both --fit-colormap and --project-colormap"
    exit 1
fi

# Print header
echo ""
echo "========================================="
echo "  Manifold Genetics Pipeline"
echo "========================================="
echo ""
echo "Mode: ${MODE:-custom}"
echo "Fit PLINK: $FIT_PLINK"
echo "Project PLINK: $PROJECT_PLINK"
echo "Output: $OUTPUT"
echo "PCs: $N_PCS"
echo "K range: $K_MIN-$K_MAX"
echo "Embedding: $EMBEDDING (input: $EMBEDDING_INPUT, knn: $KNN, t: $T)"
if [[ -n "$THREADS" ]]; then
    echo "Threads: $THREADS"
fi
if [[ -n "$NUM_GPUS" ]] && [[ "$NUM_GPUS" -gt 0 ]]; then
    echo "GPUs: $NUM_GPUS"
fi
echo ""

# Check virtual environment
print_status "Checking virtual environment..."
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_error "Virtual environment not activated"
    echo ""
    echo "Please activate the virtual environment first:"
    echo "  source .venv/bin/activate"
    echo ""
    exit 1
fi
print_success "Virtual environment active: $VIRTUAL_ENV"

# Verify required files
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
    print_error "Missing required files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

print_success "All required files found"

# Create output directory
mkdir -p "$OUTPUT"

# Build pipeline command
print_status "Running pipeline..."
echo ""

CMD="manifold-genetics pipeline"
CMD="$CMD --fit-plink \"$FIT_PLINK\""
CMD="$CMD --project-plink \"$PROJECT_PLINK\""
CMD="$CMD --fit-labels \"$FIT_LABELS\""
CMD="$CMD --project-labels \"$PROJECT_LABELS\""
CMD="$CMD --fit-colormap \"$FIT_COLORMAP\""
CMD="$CMD --project-colormap \"$PROJECT_COLORMAP\""
CMD="$CMD --output \"$OUTPUT\""
[[ -n "$GEOGRAPHIC" ]] && CMD="$CMD --geographic \"$GEOGRAPHIC\""
CMD="$CMD --n-pcs $N_PCS"
CMD="$CMD --k-min $K_MIN --k-max $K_MAX"
CMD="$CMD --embedding $EMBEDDING"
CMD="$CMD --embedding-input $EMBEDDING_INPUT"
CMD="$CMD --knn $KNN --t $T"
[[ -n "$N_LANDMARK" ]] && CMD="$CMD --n-landmark $N_LANDMARK"
[[ "$RANDOM_LANDMARKING" == "true" ]] && CMD="$CMD --random-landmarking"
[[ -n "$ADMIXTURE_GROUP_COLUMN" ]] && CMD="$CMD --admixture-group-column $ADMIXTURE_GROUP_COLUMN"
CMD="$CMD --admixture-within-group-order $ADMIXTURE_WITHIN_GROUP_ORDER"
[[ -n "$THREADS" ]] && CMD="$CMD --threads $THREADS"
[[ -n "$NUM_GPUS" ]] && CMD="$CMD --num-gpus $NUM_GPUS"
[[ -n "$NEURALADMIXTURE_BATCH_SIZE" ]] && CMD="$CMD --neuraladmixture-batch-size $NEURALADMIXTURE_BATCH_SIZE"
[[ -n "$EMBED_BATCH_SIZE" ]] && CMD="$CMD --embed-batch-size $EMBED_BATCH_SIZE"
[[ -n "$FLASHPCA_OUTPUT_DIR" ]] && CMD="$CMD --flashpca-output-dir \"$FLASHPCA_OUTPUT_DIR\""
[[ "$SKIP_METRICS" == "true" ]] && CMD="$CMD --skip-metrics"
[[ "$SKIP_PCA" == "true" ]] && CMD="$CMD --skip-pca"
[[ "$SKIP_ADMIXTURE" == "true" ]] && CMD="$CMD --skip-admixture"
[[ "$SKIP_EMBEDDING" == "true" ]] && CMD="$CMD --skip-embedding"

echo "Command:"
echo "$CMD"
echo ""

# Execute pipeline
eval $CMD

PIPELINE_EXIT_CODE=$?

# Summary
echo ""
if [[ $PIPELINE_EXIT_CODE -eq 0 ]]; then
    echo "========================================="
    print_success "Pipeline complete!"
    echo "========================================="
    echo ""
    echo "Output directory: $OUTPUT"
else
    echo "========================================="
    print_error "Pipeline failed with exit code $PIPELINE_EXIT_CODE"
    echo "========================================="
    exit $PIPELINE_EXIT_CODE
fi
