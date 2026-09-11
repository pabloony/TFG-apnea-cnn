#!/usr/bin/env bash
#SBATCH -J dataset_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --constraint=cal
#SBATCH --output=tests/job_data/graphs_phases/phase5/phase5.%j.out
#SBATCH --error=tests/job_data/graphs_phases/phase5/phase5.%j.err

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

echo "CPUs asignados: ${SLURM_CPUS_PER_TASK}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"

echo "[$(date)] Iniciando compare Phase 5..."
python -m src.graphs.compare_experiments --phase 5

echo "[$(date)] Phase 5 completado."