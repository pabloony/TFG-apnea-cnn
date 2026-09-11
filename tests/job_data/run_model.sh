#!/usr/bin/env bash
#SBATCH -J train_cnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --constraint=cal
#SBATCH --output=tests/job_data/50pc/train_cnn.%j.out
#SBATCH --error=tests/job_data/50pc/train_cnn.%j.err

set -euo pipefail

# Limitar hilos a lo asignado (evita oversubscription)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# TensorFlow CPU threads (importante)
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTEROP_THREADS=2

# Logs en tiempo real
export PYTHONUNBUFFERED=1

# (Opcional) Reduce logs de TF
export TF_CPP_MIN_LOG_LEVEL=2

# Ejecutar tu script original

time python -m src.design.training_cnn