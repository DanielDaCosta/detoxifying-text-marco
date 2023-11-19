#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
#SBATCH --account=jonmay_231
#SBATCH --mem=32G

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate marco_2

# run
cd ..
python3 finetune_bart.py --train_batch_size 32 --max_steps 50000 --save_steps 1000 --lr 1e-6 --max_source_length 180 --max_target_length 230 --train_data datasets/toxic/train_toxic.csv --val_data datasets/toxic/val_toxic.csv