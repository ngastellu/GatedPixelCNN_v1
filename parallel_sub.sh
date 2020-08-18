#!/bin/bash
#SBATCH --account=YOUR ACCOUNT
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=140G
#SBATCH --time=3-0:00
#SBATCH --mail-user=YOUR_ADDRESS@EMAIL.COM
#SBATCH --mail-type=END
#SBATCH --array=0-9

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load cuda cudnn
source ~/PixelCNN/bin/activate # initialize your virtual environment

python ./main.py --run_num=$SLURM_ARRAY_TASK_ID # run 10 jobs with parameters indexed 0-9, according to the batch_params.pkl
