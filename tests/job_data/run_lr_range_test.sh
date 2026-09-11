#!/usr/bin/env bash
##SBATCH -J lr_range_test

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --error=tests/job_data/lr_range_test/lr_range_test.%J.err
#SBATCH --output=tests/job_data/lr_range_test/lr_range_test.%J.out

echo "========== JOB INFO =========="
echo "Fecha    : $(date)"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "CPUs     : $SLURM_CPUS_PER_TASK"
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "==============================="



echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

time python -m src.design.lr_range_test \
    --batch_size 2000 \
    --num_steps  557 \
    --lr_min     1e-6 \
    --lr_max     1e-2 \
    --beta       0.9

echo "========== FIN: $(date) =========="