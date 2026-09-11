#!/usr/bin/env bash
#SBATCH -J train_val_cpu
#SBATCH --partition=all_nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --constraint=cal
#SBATCH --output=tests/job_data/train_val/train_val_cpu.%j.out
#SBATCH --error=tests/job_data/train_val/train_val_cpu.%j.err

set -euo pipefail

# -----------------------------
# Forzar CPU y evitar warnings CUDA/cuDNN/cuBLAS
# -----------------------------

export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_FLAGS=--xla_gpu_cuda_data_dir=


# -----------------------------
# Threads (evitar oversubscription)
# -----------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# TensorFlow threading
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTEROP_THREADS=2

# Logs
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

echo "========== JOB INFO =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "==============================="

time srun -u python -m src.design.main_train_val
