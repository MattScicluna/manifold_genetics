#!/bin/bash
#
# Submit the real-cohort test suite to SLURM.
#
# The cohort pipeline tests run the shipped configs against real data, which for
# the controlled-access cohorts means hours and hundreds of gigabytes -- more
# than a login node or an interactive allocation should carry.
#
# Usage:
#   bash tests/integration/submit_cohort_tests.sh <cohort> [options]
#
# Examples:
#   bash tests/integration/submit_cohort_tests.sh hgdp
#   bash tests/integration/submit_cohort_tests.sh ukbb_projection --mem=256GB --time=24:00:00
#   bash tests/integration/submit_cohort_tests.sh all --admixture --gpus=1
#
# Cohort names come from tests/integration/cohorts.py:
#   hgdp  ukbb_projection  ukbb_subsample  ukbb_geosketch
#   aou_projection  aou_subsample  aou_geosketch  all
#
# Options:
#   --cpus=N         CPUs (default: 8)
#   --mem=SIZE       Memory (default: 64GB; the 486k-sample cohorts want more)
#   --time=HH:MM:SS  Walltime (default: 12:00:00)
#   --gpus=N         GPUs (default: 0; only useful with --admixture)
#   --account=NAME   SLURM account (default: $SLURM_ACCOUNT)
#   --admixture      Include neural admixture; off by default
#
# Run the preflight checks first -- they take seconds and catch the
# data/config disagreements that would otherwise surface at the end of this:
#
#   pytest tests/integration/test_cohort_preflight.py -v

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${MANIFOLD_GENETICS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ACCOUNT="${SLURM_ACCOUNT:-}"

if [ $# -lt 1 ]; then
    sed -n '2,31p' "${BASH_SOURCE[0]}" | sed 's/^#//'
    exit 1
fi

COHORT="$1"
shift

CPUS=8
MEM="64GB"
TIME="12:00:00"
GPUS=0
ADMIXTURE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --cpus=*)    CPUS="${1#*=}" ;;
        --mem=*)     MEM="${1#*=}" ;;
        --time=*)    TIME="${1#*=}" ;;
        --gpus=*)    GPUS="${1#*=}" ;;
        --account=*) ACCOUNT="${1#*=}" ;;
        --admixture) ADMIXTURE="--cohort-admixture" ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

if [ -z "$ACCOUNT" ]; then
    echo "Error: no SLURM account. Set \$SLURM_ACCOUNT or pass --account=NAME" >&2
    exit 1
fi

JOB_NAME="cohort_${COHORT}"
mkdir -p "${PROJECT_ROOT}/logs"

GPU_LINE=""
[ "$GPUS" -gt 0 ] && GPU_LINE="#SBATCH --gres=gpu:${GPUS}"

BATCH=$(mktemp /tmp/manifold_cohort_XXXXXX.sh)
cat > "$BATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --time=${TIME}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --output=${PROJECT_ROOT}/logs/${JOB_NAME}_%j.out
#SBATCH --error=${PROJECT_ROOT}/logs/${JOB_NAME}_%j.err
${GPU_LINE}

set -e
cd ${PROJECT_ROOT}
source .venv/bin/activate

echo "Cohort: ${COHORT}   Node: \$SLURMD_NODENAME   Started: \$(date)"

# Preflight first: seconds, and it fails loudly on the data/config mismatches
# that would otherwise be discovered after the expensive part.
python -m pytest tests/integration/test_cohort_preflight.py -v

python -m pytest tests/integration/test_cohort_pipeline_real.py \\
    -v -s -m "slow and integration" --cohort ${COHORT} ${ADMIXTURE}

echo "Finished: \$(date)"
EOF

echo "Submitting ${JOB_NAME}: ${CPUS} CPUs, ${MEM}, ${TIME}, ${GPUS} GPU(s), account ${ACCOUNT}"
JOB_ID=$(sbatch --parsable "$BATCH")
rm -f "$BATCH"

echo "Job ID: ${JOB_ID}"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f ${PROJECT_ROOT}/logs/${JOB_NAME}_${JOB_ID}.out"
