#!/bin/bash
#
# Cluster Detection Script
#
# Detects the current cluster environment (Narval, Mila, AoU, or local)
# and sets normalized environment variables for CPUs and GPUs.
#
# Sets the following environment variables:
#   CLUSTER_NAME  - Name of the cluster (narval, mila, aou, local)
#   CLUSTER_CPUS  - Number of CPUs allocated to the job
#   CLUSTER_GPUS  - Number of GPUs allocated to the job (0 if none)
#
# Usage:
#   source examples/_shared/detect_cluster.sh
#   echo "Running on $CLUSTER_NAME with $CLUSTER_CPUS CPUs and $CLUSTER_GPUS GPUs"
#

detect_cluster() {
    # Detect cluster by hostname or environment variables
    if hostname | grep -q "narval"; then
        CLUSTER_NAME="narval"
    elif hostname | grep -q "mila.quebec"; then
        CLUSTER_NAME="mila"
    elif [[ -n "$WORKSPACE_CDR" ]]; then
        CLUSTER_NAME="aou"
    else
        CLUSTER_NAME="local"
    fi

    # Detect CPUs (try multiple SLURM variables)
    if [[ -n "$SLURM_CPUS_PER_TASK" ]]; then
        CLUSTER_CPUS="$SLURM_CPUS_PER_TASK"
    elif [[ -n "$SLURM_CPUS_ON_NODE" ]]; then
        CLUSTER_CPUS="$SLURM_CPUS_ON_NODE"
    elif [[ -n "$SLURM_JOB_CPUS_PER_NODE" ]]; then
        CLUSTER_CPUS="$SLURM_JOB_CPUS_PER_NODE"
    else
        # Fall back to OS detection
        if command -v nproc &> /dev/null; then
            CLUSTER_CPUS=$(nproc)
        else
            CLUSTER_CPUS=4  # Safe default
        fi
    fi

    # Detect GPUs (prioritize allocation over hardware)
    CLUSTER_GPUS=0

    # 1. Trust CUDA_VISIBLE_DEVICES if set (standard for controlling access)
    if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
        if [[ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]]; then
            CLUSTER_GPUS=0
        else
            # Count comma-separated devices
            CLUSTER_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
        fi
    # 2. Try SLURM allocation variables (specific to the job/task)
    elif [[ -n "$SLURM_GPUS_PER_TASK" ]]; then
        CLUSTER_GPUS="$SLURM_GPUS_PER_TASK"
    elif [[ -n "$SLURM_GPUS" ]]; then
        CLUSTER_GPUS="$SLURM_GPUS"
    elif [[ -n "$SLURM_JOB_GPUS" ]]; then
        # SLURM_JOB_GPUS is often a list of IDs, count them
        CLUSTER_GPUS=$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | wc -l)
    # 3. Fall back to nvidia-smi (mostly for local machines)
    elif command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
        if [[ $GPU_COUNT -gt 0 ]]; then
            CLUSTER_GPUS=$GPU_COUNT
        fi
    fi

    export CLUSTER_NAME CLUSTER_CPUS CLUSTER_GPUS
}

# Auto-execute on source
detect_cluster

# Print detection results if VERBOSE is set
if [[ -n "$VERBOSE" ]]; then
    echo "Cluster detection:"
    echo "  CLUSTER_NAME: $CLUSTER_NAME"
    echo "  CLUSTER_CPUS: $CLUSTER_CPUS"
    echo "  CLUSTER_GPUS: $CLUSTER_GPUS"
fi
