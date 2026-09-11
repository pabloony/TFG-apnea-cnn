#!/usr/bin/env bash
#SBATCH -J eval_val
#SBATCH --partition=all_nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --constraint=cal
#SBATCH --output=tests/job_data/w15/train/t050_bs2000_d35_rlrp_w15_pclip.out
#SBATCH --error=tests/job_data/w15/train/t050_bs2000_d35_rlrp_w15_pclip.err

set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

time srun -u python -m src.design.eval_val
