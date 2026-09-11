#!/usr/bin/env bash
#SBATCH -J preprocess_apnea_w15
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --constraint=cal
#SBATCH --output=tests/job_data/w15/preprocessingSTLK.%j.out
#SBATCH --error=tests/job_data/w15/preprocessingSTLK.%j.err

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

#echo "[$(date)] Iniciando resample..."
#time python -m src.preprocessing.main_resample

#echo "[$(date)] Iniciando mask (sueño)..."
#time python -m src.preprocessing.main_mask

#echo "[$(date)] Iniciando mask_events..."
#time python -m src.preprocessing.main_mask_events

#echo "[$(date)] Iniciando delay..."
#time python -m src.preprocessing.main_delay_extra

echo "[$(date)] Iniciando labels..."
time python -m src.preprocessing.main_labels

echo "[$(date)] Iniciando normalisation2..."
time python -m src.preprocessing.main_normalisation2

echo "[$(date)] Preprocessing completado."