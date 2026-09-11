#!/bin/bash
#
# Batch submission wrapper for manifold-genetics examples
#
# Written for SLURM clusters. The repo root is derived from this script's own
# location; the account comes from $SLURM_ACCOUNT (or --account=NAME), so nothing
# is tied to one cluster.
#
# Usage:
#   bash examples/_shared/submit_batch.sh <script_path> [SLURM_options] [-- pipeline_options]
#
# Examples:
#   bash examples/_shared/submit_batch.sh examples/hgdp_1kgp/run_pipeline.sh
#   bash examples/_shared/submit_batch.sh examples/ukbb/10k_WB_5K_Irish/run_pipeline.sh --cpus=16 --mem=128GB
#   bash examples/_shared/submit_batch.sh examples/ukbb/hgdp_1kgp_proj/run_pipeline.sh -- --skip-admixture
#   bash examples/_shared/submit_batch.sh examples/aou/60k_white/run_pipeline.sh --time=48:00:00 --job=my_job -- --skip-pca
#
# SLURM Options (before --):
#   --cpus=N         Number of CPUs (default: 8)
#   --mem=SIZE       Memory allocation (default: 32GB)
#   --time=HH:MM:SS  Time limit (default: 24:00:00)
#   --job=NAME       Job name (default: directory name)
#   --gpus=N         Number of GPUs (default: 0, loads cuda module if >0)
#   --account=NAME   SLURM account (default: $SLURM_ACCOUNT)
#
# Pipeline Options (after --):
#   Any arguments after -- are passed directly to the pipeline script
#

set -e

# Repo root = two levels up from examples/_shared/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${MANIFOLD_GENETICS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ACCOUNT="${SLURM_ACCOUNT:-}"

# Check if script path is provided
if [ $# -lt 1 ]; then
    echo "Error: script path required"
    echo ""
    echo "Usage: bash examples/_shared/submit_batch.sh <script_path> [SLURM_options] [-- pipeline_options]"
    echo ""
    echo "Examples:"
    echo "  bash examples/_shared/submit_batch.sh examples/hgdp_1kgp/run_pipeline.sh"
    echo "  bash examples/_shared/submit_batch.sh examples/ukbb/10k_WB_5K_Irish/run_pipeline.sh --cpus=16 --mem=128GB"
    echo "  bash examples/_shared/submit_batch.sh examples/ukbb/hgdp_1kgp_proj/run_pipeline.sh -- --skip-admixture"
    echo ""
    exit 1
fi

# Get script path (first argument)
RUN_SCRIPT="$1"
shift

# Separate SLURM options from pipeline options (split on --)
SLURM_ARGS=()
PIPELINE_ARGS=()
FOUND_SEPARATOR=false

for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
        FOUND_SEPARATOR=true
    elif [[ "$FOUND_SEPARATOR" == true ]]; then
        PIPELINE_ARGS+=("$arg")
    else
        SLURM_ARGS+=("$arg")
    fi
done

# Reset positional parameters to SLURM args only
set -- "${SLURM_ARGS[@]}"

# Resolve to absolute path relative to PROJECT_ROOT
if [[ "$RUN_SCRIPT" = /* ]]; then
    # Already absolute
    RUN_SCRIPT_ABS="$RUN_SCRIPT"
else
    # Make absolute relative to PROJECT_ROOT
    RUN_SCRIPT_ABS="${PROJECT_ROOT}/${RUN_SCRIPT}"
fi

# Check if run_pipeline.sh exists
if [ ! -f "$RUN_SCRIPT_ABS" ]; then
    echo "Error: run_pipeline.sh not found: $RUN_SCRIPT_ABS"
    exit 1
fi

# Get example directory
EXAMPLE_DIR="$(dirname "$RUN_SCRIPT_ABS")"

# Default SLURM parameters
CPUS=8
MEM="32GB"
TIME="24:00:00"
GPUS=0

# Derive job name from example directory if not provided
EXAMPLE_NAME=$(basename "$EXAMPLE_DIR")
JOB_NAME="$EXAMPLE_NAME"

# Parse SLURM arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --cpus=*)
            CPUS="${1#*=}"
            ;;
        --mem=*)
            MEM="${1#*=}"
            ;;
        --time=*)
            TIME="${1#*=}"
            ;;
        --job=*)
            JOB_NAME="${1#*=}"
            ;;
        --gpus=*)
            GPUS="${1#*=}"
            ;;
        --account=*)
            ACCOUNT="${1#*=}"
            ;;
        *)
            echo "Warning: Unknown SLURM option: $1 (use -- to pass options to pipeline)"
            ;;
    esac
    shift
done

if [ -z "$ACCOUNT" ]; then
    echo "Error: no SLURM account. Set \$SLURM_ACCOUNT or pass --account=NAME" >&2
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/logs"

# Convert pipeline args array to string for embedding in heredoc
PIPELINE_ARGS_STR="${PIPELINE_ARGS[*]}"

# Create temporary batch script
TEMP_BATCH_SCRIPT=$(mktemp /tmp/manifold_batch_XXXXXX.sh)

# Build GPU SBATCH line if needed
if [ "$GPUS" -gt 0 ]; then
    GPU_SBATCH_LINE="#SBATCH --gres=gpu:${GPUS}"
else
    GPU_SBATCH_LINE=""
fi

cat > "$TEMP_BATCH_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --time=${TIME}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --output=${PROJECT_ROOT}/logs/${JOB_NAME}_%j.out
#SBATCH --error=${PROJECT_ROOT}/logs/${JOB_NAME}_%j.err
${GPU_SBATCH_LINE}

set -e

echo "=========================================="
echo "  Manifold Genetics Pipeline: ${JOB_NAME}"
echo "=========================================="
echo ""
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "CPUs: ${CPUS}"
echo "Memory: ${MEM}"
echo "GPUs: ${GPUS}"
echo "Time limit: ${TIME}"
echo "Started: \$(date)"
echo ""

# Navigate to project root
cd ${PROJECT_ROOT}

# Create logs directory if it doesn't exist (in case running on compute node)
mkdir -p logs

# Load CUDA module if GPUs requested
if [ ${GPUS} -gt 0 ]; then
    echo "Loading CUDA modules..."
    module load StdEnv/2020 gcc/11.3.0 cuda/11.8.0
    echo "✓ CUDA loaded: \$CUDA_HOME"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Run the example pipeline
echo "Running: ${RUN_SCRIPT_ABS} ${PIPELINE_ARGS_STR}"
echo ""
bash ${RUN_SCRIPT_ABS} ${PIPELINE_ARGS_STR}

echo ""
echo "=========================================="
echo "  Pipeline Complete"
echo "=========================================="
echo "Finished: \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo ""
echo "Outputs saved to: ${EXAMPLE_DIR}/outputs/"
echo "Logs saved to: ${PROJECT_ROOT}/logs/${JOB_NAME}_\${SLURM_JOB_ID}.{out,err}"
echo ""
EOF

# Make the temp script executable
chmod +x "$TEMP_BATCH_SCRIPT"

# Show what we're about to submit
echo "=========================================="
echo "  Submitting Batch Job"
echo "=========================================="
echo ""
echo "Script: $RUN_SCRIPT_ABS"
if [ -n "$PIPELINE_ARGS_STR" ]; then
    echo "Pipeline args: $PIPELINE_ARGS_STR"
fi
echo "Job name: $JOB_NAME"
echo "CPUs: $CPUS"
echo "Memory: $MEM"
echo "GPUs: $GPUS"
echo "Time limit: $TIME"
echo "Account: $ACCOUNT"
echo ""

# Submit the job
echo "Submitting to SLURM..."
JOB_ID=$(sbatch --parsable "$TEMP_BATCH_SCRIPT")

echo ""
echo "✓ Job submitted successfully!"
echo ""
echo "Job ID: $JOB_ID"
echo "Monitor with: squeue -j $JOB_ID"
echo "Cancel with: scancel $JOB_ID"
echo "View output: tail -f ${PROJECT_ROOT}/logs/${JOB_NAME}_${JOB_ID}.out"
echo "View errors: tail -f ${PROJECT_ROOT}/logs/${JOB_NAME}_${JOB_ID}.err"
echo ""

# Clean up temp script
rm -f "$TEMP_BATCH_SCRIPT"
