#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --account=ctb-hussinju
#SBATCH --time=6:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=ukbb_hgdp_proj
#SBATCH --output=/lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics/logs/ukbb_hgdp_proj_%j.out
#SBATCH --error=/lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics/logs/ukbb_hgdp_proj_%j.err

# Batch script to run UKBB-HGDP cross-projection pipeline on Narval cluster
# Usage: sbatch examples/ukbb/hgdp_1kgp_proj/run_pipeline_batch.sh

set -e

echo "=========================================="
echo "  UKBB-HGDP Cross-Projection Pipeline"
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

# Run cross-projection pipeline
echo "Running UKBB-HGDP cross-projection pipeline..."
echo ""
bash examples/ukbb/hgdp_1kgp_proj/run_pipeline.sh

echo ""
echo "=========================================="
echo "  Cross-Projection Pipeline Complete"
echo "=========================================="
echo "Finished: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "Outputs saved to: examples/ukbb/hgdp_1kgp_proj/outputs/"
echo "Logs saved to: logs/ukbb_hgdp_proj_${SLURM_JOB_ID}.{out,err}"
echo ""
echo "Analysis type: Cross-cohort projection (HGDP reference → UKBB target)"
echo "Key features:"
echo "  • SNP intersection between UKBB and HGDP datasets"
echo "  • HGDP populations as diverse reference for model training"
echo "  • UKBB samples projected into HGDP-trained space"
echo "  • Separate population visualizations for each cohort"
echo ""