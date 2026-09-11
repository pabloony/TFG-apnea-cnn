#!/usr/bin/env bash
#SBATCH -J dta_stnf_all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --constraint=cal
#SBATCH --output=tests/job_data/50pc_c/imput_data_trainval.%j.out
#SBATCH --error=tests/job_data/50pc_c/imput_data_trainval.%j.err

set -euo pipefail





# Limitar hilos de BLAS/OpenMP a los CPUs asignados
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Logs en tiempo real
export PYTHONUNBUFFERED=1

# Ejecutar
time python -m src.complementation.main_imput_trainval_v2
