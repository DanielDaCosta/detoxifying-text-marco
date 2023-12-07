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
conda activate marco_3

# run
cd ..
python3 finetune_bart.py --tok_type lucadiliello/bart-small --model_type lucadiliello/bart-small --train_batch_size 32 --eval_batch_size 128 --max_steps 25000 --save_steps 1000 --lr 5e-7 --max_source_length 180 --max_target_length 230 --train_data datasets/toxic/train_toxic.csv --val_data datasets/toxic/val_toxic.csv