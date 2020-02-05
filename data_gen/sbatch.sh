#!/bin/bash
#SBATCH -N 1 -n 1 -c 4 --mem-per-cpu=24000 -t 1-00:00:00

module purge
module load gcc/5.4.0 python-env/intelpython3.6-2018.3
module load openmpi/2.1.2 cuda/9.0 cudnn/7.4.1-cuda9

srun python gen_motion_data.py
