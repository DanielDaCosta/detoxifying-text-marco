# CARC Outputs

## Expert model
Google Drive: https://drive.google.com/drive/folders/1RAuzXQmb8n1ILaY4noV3hRdqm_yjeX1i?usp=drive_link

### Files
- `expert-model/expert-fine-tuned.out`: CARC output fille
- `expert-model/bart-base_2e-06_0_96_jigsaw_full_30/`: fine-tuned model on Toxic dataset.
- `expert-model/expert.slurm`: CARC job script

### Dataset
- `train_non_toxic.csv` & `val_non_toxic.csv`: https://drive.google.com/drive/folders/1TeUC1swuEIScsIzI2Cnl28Yg3fF8vfzi?usp=sharing

## Anti-expert-model

Google Drive path: https://drive.google.com/drive/folders/1smVZ0xouuVVNZE6pbmgQF6z48yEEBagk?usp=sharing

### Files
- `anti-expert-model/anti-expert-fine-tuned.out`: CARC output fille
- `anti-expert-model/bart-base_1e-06_0_32_jigsaw_full_30/`: fine-tuned model on Toxic dataset.
- `anti-expert-model/anti-expert.slurm`: CARC job script


### Dataset
- `train_toxic.csv` & `val_toxic.csv`: https://drive.google.com/drive/folders/1TeUC1swuEIScsIzI2Cnl28Yg3fF8vfzi?usp=sharing