# CARC Outputs

## Expert model
Google Drive(<ins>You must use your USC account to access</ins>): https://drive.google.com/drive/folders/1smVZ0xouuVVNZE6pbmgQF6z48yEEBagk?usp=sharing

### Files
- `expert-model/expert-fine-tuned.out`: CARC output fille
- `expert-model/bart-base_2e-06_0_96_jigsaw_full_30/`: fine-tuned model on Toxic dataset.
- `expert-model/expert.slurm`: CARC job script
- `expert-model/expert-bart-small.slurm`: CARC job script for BART small model
- `expert-model/bart-small_1.4e-06_0_96_jigsaw_full_30`: fine-tined bart-small model on toxic dataset
- `expert-model/expert-fine-tuned.out`: CARC output file for BART-small

### Dataset
- `train_non_toxic.csv` & `val_non_toxic.csv`: https://drive.google.com/drive/folders/1TeUC1swuEIScsIzI2Cnl28Yg3fF8vfzi?usp=sharing

## Anti-expert-model

Google Drive path(<ins>You must use your USC account to access</ins>): https://drive.google.com/drive/folders/1RAuzXQmb8n1ILaY4noV3hRdqm_yjeX1i?usp=sharing

### Files
- `anti-expert-model/anti-expert-fine-tuned.out`: CARC output fille
- `anti-expert-model/bart-base_1e-06_0_32_jigsaw_full_30/`: fine-tuned model on Toxic dataset.
- `anti-expert-model/anti-expert.slurm`: CARC job script
- `anti-expert-model/bart-small/anti-expert-bart-small.slurm`: CARC job script for BART small model
- `anti-expert-model/bart-small/bart-small_7e-07_0_32_jigsaw_full_30/`: fine-tined bart-small model on toxic dataset
- `anti-expert-model/bart-small/anti-expert-bart-small-output.out`: CARC output file for BART-small


### Dataset
- `train_toxic.csv` & `val_toxic.csv`: https://drive.google.com/drive/folders/1TeUC1swuEIScsIzI2Cnl28Yg3fF8vfzi?usp=sharing