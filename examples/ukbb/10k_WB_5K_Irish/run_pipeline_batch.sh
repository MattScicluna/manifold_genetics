#!/bin/bash
#SBATCH --job-name=ukbb_10k_wb_5k_irish
#SBATCH --account=ctb-hussinju
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --output=/lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics/logs/ukbb_10k_wb_5k_irish_%j.out
#SBATCH --error=/lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics/logs/ukbb_10k_wb_5k_irish_%j.err

# Absolute paths (SLURM copies script to temp location, so relative paths don't work)
PROJECT_ROOT="/lustre06/project/6065672/sciclun4/ActiveProjects/manifold_genetics"
SCRIPT_DIR="${PROJECT_ROOT}/examples/ukbb/10k_WB_5K_Irish"

# Load CUDA module for GPU support
module load cuda/12.2

# Activate virtual environment
source ${PROJECT_ROOT}/.venv/bin/activate

# Run pipeline
bash ${SCRIPT_DIR}/run_pipeline.sh
