# Model Training 

Code to finetune the BART model is located in `finetune_bart.py`. 

# Base Model

## Anti-Expert Model
BART-base further finetuned on the toxic portion of the Jigsaw Corpus with the same pretraining masked denoising objective.

To run, move the `finetune_bart.py` file to the root directory and run:

    python3 finetune_bart.py --train_batch_size 32 --max_steps 50000 --save_steps 1000 --lr 1e-6 --max_source_length 180 --max_target_length 230 --train_data datasets/toxic/train_toxic.csv --val_data datasets/toxic/val_toxic.csv

The model was fine-tuned on CARC using A100 GPUs, with the following configuration. Please refer to *CARC/README.md*:

```bash
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
python3 finetune_bart.py --train_batch_size 32 --max_steps 50000 --save_steps 1000 --lr 1e-6 --max_source_length 180 --max_target_length 230 --train_data datasets/toxic/train_toxic.csv --val_data datasets/toxic/val_toxic.csv
```

## Expert Model
BART-base further finetuned on the ***non-toxic*** portion of the Jigsaw Corpus with the same pretraining masked denoising objective


To run, move the `finetune_bart.py` file to the root directory and run:

    python3 finetune_bart.py --train_batch_size 48 --eval_batch_size 96 --max_steps 100000 --save_steps 5000 --lr 2e-6 --max_source_length 180 --max_target_length 230 --train_data datasets/toxic/train_non_toxic.csv --val_data datasets/toxic/val_non_toxic.csv

The model was fine-tuned on CARC using 2xA100 GPUs, with the following configuration. Please refer to *CARC/README.md*:

```
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
python3 finetune_bart.py --train_batch_size 48 --eval_batch_size 96 --max_steps 100000 --save_steps 5000 --lr 2e-6 --max_source_length 180 --max_target_length 230 --train_data datasets/toxic/train_non_toxic.csv --val_data datasets/toxic/val_non_toxic.csv
```

# BART-Small
For BART-Small we used the BART model hosted on Hugging Face:  https://huggingface.co/lucadiliello/bart-small


The procedure for running the scripts remains consistent with the previously discussed method, albeit utilizing distinct parameters.

The batch jobs scripts used for fine-tuning are located in the `CARC/` folder.

## Anti-Expert
To run, move the `finetune_bart.py` file to the root directory and run:

    python3 finetune_bart.py --tok_type lucadiliello/bart-small --model_type lucadiliello/bart-small --train_batch_size 32 --eval_batch_size 128 --max_steps 25000 --save_steps 1000 --lr 5e-7 --max_source_length 180 --max_target_length 230 --train_data datasets/toxic/train_toxic.csv --val_data datasets/toxic/val_toxic.csv

## Expert

    python3 finetune_bart.py --tok_type lucadiliello/bart-small --model_type lucadiliello/bart-small --train_batch_size 48 --eval_batch_size 96 --max_steps 50000 --save_steps 2500 --lr 1.5e-6 --max_source_length 180 --max_target_length 230 --train_data datasets/toxic/train_non_toxic.csv --val_data datasets/toxic/val_non_toxic.csv