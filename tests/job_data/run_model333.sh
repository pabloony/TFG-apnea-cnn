#!/usr/bin/env bash
#SBATCH -J train_cnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G            # He bajado a 32G. Para CNN y batch 250, 64G suele sobrar mucho.
#SBATCH --time=04:00:00      # He bajado el tiempo. Si tu entrenamiento dura menos, pon menos.
#SBATCH --output=tests/job_data/333/train_cnn333_bs:2000.%j.out
#SBATCH --error=tests/job_data/333/train_cnn333_bs:2000.%j.err
set -euo pipefail

# Configuración eficiente para CPU
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTEROP_THREADS=1
export CUDA_VISIBLE_DEVICES="" # Asegura que no intente usar GPU

# Ejecutar script
time python -m src.design.training_cnn333