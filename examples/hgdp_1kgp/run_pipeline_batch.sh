#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --account=ctb-hussinju
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=hgdp_1kgp
#SBATCH --output=/lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics/logs/hgdp_1kgp_%j.out
#SBATCH --error=/lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics/logs/hgdp_1kgp_%j.err

# Batch script to run HGDP+1KGP example pipeline on Narval cluster
# Usage: sbatch examples/hgdp_1kgp/run_pipeline_batch.sh

set -e

echo "=========================================="
echo "  HGDP+1KGP Example Pipeline"
echo "=========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo ""

# Navigate to project directory
cd /lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Run pipeline (handles data download and preparation automatically)
echo "Running HGDP+1KGP pipeline..."
echo ""
bash examples/hgdp_1kgp/run_pipeline.sh

echo ""
echo "=========================================="
echo "  Pipeline Complete"
echo "=========================================="
echo "Finished: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "Outputs saved to: examples/hgdp_1kgp/outputs/"
echo "Logs saved to: logs/hgdp_1kgp_${SLURM_JOB_ID}.{out,err}"
