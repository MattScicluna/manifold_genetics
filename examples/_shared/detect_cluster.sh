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

    # Detect GPUs (try multiple SLURM variable names across clusters)
    CLUSTER_GPUS=0

    if [[ -n "$SLURM_GPUS" ]]; then
        CLUSTER_GPUS="$SLURM_GPUS"
    elif [[ -n "$SLURM_GPUS_ON_NODE" ]]; then
        CLUSTER_GPUS="$SLURM_GPUS_ON_NODE"
    elif [[ -n "$SLURM_GPUS_PER_NODE" ]]; then
        CLUSTER_GPUS="$SLURM_GPUS_PER_NODE"
    elif [[ -n "$SLURM_JOB_GPUS" ]]; then
        CLUSTER_GPUS="$SLURM_JOB_GPUS"
    else
        # Fall back to nvidia-smi if available
        if command -v nvidia-smi &> /dev/null; then
            GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
            if [[ $GPU_COUNT -gt 0 ]]; then
                CLUSTER_GPUS=$GPU_COUNT
            fi
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
