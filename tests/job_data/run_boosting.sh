#!/usr/bin/env bash
#SBATCH -J boosting_round1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=dgx
#SBATCH --error=tests/job_data/balanceo_alternativo/boosting/boosting_round1.%J.err
#SBATCH --output=tests/job_data/balanceo_alternativo/boosting/boosting_round1.%J.out

echo "========== JOB INFO =========="
echo "Fecha    : $(date)"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "CPUs     : $SLURM_CPUS_PER_TASK"
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "==============================="

source activate oxinet_env

cd /mnt/home/users/ac_aux/portega/ProyectoPython3.0

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

time python -m src.design.boosting_round1

echo "========== FIN: $(date) =========="

