#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --account=jonmay_231

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate marco

# run
cd ..
python3 finetune_bart.py --max_steps 2