#!/usr/bin/env bash
#SBATCH -J complementation_apnea_w15
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G                  # X_train solo pesa ~1.4M x 2 x 500 x 4bytes ≈ 5.6GB
#SBATCH --time=03:00:00            # pero durante construcción el pico puede ser el doble
#SBATCH --constraint=cal
#SBATCH --output=tests/job_data/w15/complementation_STLK.%j.out
#SBATCH --error=tests/job_data/w15/complementation_STLK.%j.err

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

echo "[$(date)] Iniciando joining..."
time python -m src.complementation.main_joining

echo "[$(date)] Iniciando imput_trainval..."
time python -m src.complementation.main_imput_trainval_v2

echo "[$(date)] Complementation completado."