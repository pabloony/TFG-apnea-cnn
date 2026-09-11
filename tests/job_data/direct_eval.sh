#!/usr/bin/env bash
#SBATCH -J              eval_piorecky_direct
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --constraint=cal
#SBATCH --output=tests/job_data/direct_eval/eval_direct.%j.out
#SBATCH --error=tests/job_data/direct_eval/eval_direct.%j.err

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2



python -m src.design.eval_piorecky_direct