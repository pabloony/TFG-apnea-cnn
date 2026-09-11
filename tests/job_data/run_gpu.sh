#!/usr/bin/env bash
#SBATCH -J train_val_cnn
#SBATCH --partition=all_nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=tests/job_data/train_val/train_val.%j.out
#SBATCH --error=tests/job_data/train_val/train_val.%j.err

set -euo pipefail

# -----------------------------
# CPU threads (alimentan la GPU)
# -----------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTEROP_THREADS=1

# -----------------------------
# Logs limpios
# -----------------------------
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

# No reservar toda la VRAM de golpe
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "========== JOB INFO =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "==============================="

time srun -u python -m src.design.main_train_val
