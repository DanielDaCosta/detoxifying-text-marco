#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
#SBATCH --account=jonmay_231
#SBATCH --mem=32G

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate marco_3

# run
cd ..
python3 finetune_bart.py --tok_type lucadiliello/bart-small --model_type lucadiliello/bart-small --train_batch_size 48 --eval_batch_size 96 --max_steps 50000 --save_steps 2500 --lr 1.5e-6 --max_source_length 180 --max_target_length 230 --train_data datasets/toxic/train_non_toxic.csv --val_data datasets/toxic/val_non_toxic.csv