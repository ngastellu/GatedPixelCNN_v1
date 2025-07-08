#!/bin/bash
#SBATCH --account=YOUR ACCOUNT
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=140G
#SBATCH --time=3-0:00
#SBATCH --mail-user=YOUR_ADDRESS@EMAIL.COM
#SBATCH --mail-type=END
#SBATCH --array=0-1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load cuda cudnn
source ~/PixelCNN/bin/activate # initialize your virtual environment

gnrs=('armchair' 'zigzag')
python ./main.py ${gnrs[$SLURM_ARRAY_TASK_ID]}