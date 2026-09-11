#!/usr/bin/env bash
#SBATCH -J val_probs_gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=dgx
#SBATCH --output=tests/job_data/val_probs_per_subject/val_probs_STLK.out
#SBATCH --error=tests/job_data/val_probs_per_subject/val_probs_STLK.err
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

time srun -u python -m src.design.export_stlk_probs_per_subject 

echo "========== FIN: $(date) =========="
