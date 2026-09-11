#!/usr/bin/env bash
#SBATCH -J val_probs
#SBATCH --partition=all_nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --constraint=cal
#SBATCH --output=tests/job_data/val_probs_per_subject/val_probs_p1.out
#SBATCH --error=tests/job_data/val_probs_per_subject/val_probs_p1.err

set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

time srun -u python -m src.design.export_val_probs_per_subject
