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
python finetune_t5.py --train_batch_size 32 --max_steps 50000 --save_steps 1000 --lr 1e-6 --max_source_length 180 --max_target_length 230 --train_data dataset/train/train_toxic.csv --val_data dataset/train/val_toxic.csv --model_dir CARC/anti-expert-model/t5