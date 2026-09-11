#!/usr/bin/env bash
#SBATCH -J train_cpu_bs1000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=48:00:00
#SBATCH --constraint=cal
#SBATCH --error=tests/job_data/t050/train_results_phase2/t050_bs1000.%J.err
#SBATCH --output=tests/job_data/t050/train_results_phase2/t050_bs1000.%J.out

echo "========== JOB INFO =========="
echo "Fecha    : $(date)"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "CPUs     : $SLURM_CPUS_PER_TASK"
echo "==============================="

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

source activate oxinet_env

cd /mnt/home/users/ac_aux/portega/ProyectoPython3.0

time python -m src.design.new_training_cnn

echo "========== FIN: $(date) =========="