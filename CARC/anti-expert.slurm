#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=15:00:00
#SBATCH --account=jonmay_231

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate marco

# run
cd ..
python3 finetune_bart.py --train_batch_size 32 --max_steps 50000 --save_steps 1000 --lr 1e-6 --max_source_length 180 --max_target_length 230