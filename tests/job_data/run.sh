#!/usr/bin/env bash
# Leave only one comment symbol on selected options
# Those with two commets will be ignored:
# The name to show in queue lists for this job:
##SBATCH -J pruebas.sh

# Number of desired cpus (can be in any node):
#SBATCH --ntasks=1

# Number of desired cpus (all in same node):
##SBATCH --cpus-per-task=4

# Amount of RAM needed for this job:
#SBATCH --mem=16gb

# The available nodes are: 
#     AMD nodes with 256 cores and 683GB of usable RAM
#     AMD nodes with 128 cores and 1800GB of usable RAM
#     AMD nodes  with 128 cores and 439GB of usable RAM
#     Intel nodes with 52  cores and 187GB of usable RAM
 
# The time the job will be running:
#SBATCH --time=10:00:00

# If you need nodes with special features you can select a constraint.
# Please, use cal by default. You will be assigned a node that satisfies your requests.
#SBATCH --constraint=cal
 
# Change "cal" by "intel" if you want to use Intel nodes and by "amd" if you want to use AMD nodes.
##SBATCH --constraint=intel
##SBATCH --constraint=amd

# To use GPU, comment out the previous constraint line and uncomment these two following lines.
##SBATCH --constraint=dgx
##SBATCH --gres=gpu:1

# Set output and error files
#SBATCH --error=tests/job_data/inspections/inspect.%j.err
#SBATCH --output=tests/job_data/inspections/inspect.%j.out

# Leave one comment in following line to make an array job. Then N jobs will be launched. In each one SLURM_ARRAY_TASK_ID will take one value from 1 to 100
##SBATCH --array=1-100

# To load some software (you can show the list with 'module avail'):
# module load software
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4


# the program to execute with its parameters:

python -m src.dta.inspect_channels_edf --n 5 --seed 42








