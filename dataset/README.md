# Dataset & Preprocess

# Training
Jigsaw dataset, split into toxic and non-toxic subsets. Toxic subset consists of data where >0.5 proportion of annotators rated toxic, while non-toxic subset consists of data where 0 proportion of annotators rated toxic. `split_dataset_code.ipynb` split the raw dataset into toxic and non-toxic data.

### Raw Dataset

Download dataset from:
- https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data?select=train.csv

### Preprocess

For preprocessing, we initially calculate the percentage of annotators who labeled each comment as toxic. Comments with annotations surpassing the 50% threshold are categorized as **toxic**. Conversely, comments not classified as toxic by any annotator are classified as **non-toxic**.

Due to the straightforward nature of this preprocessing code, it was executed in the notebook named `split_dataset_code.ipynb`.

### Final Dataset


|   # rows   | Train   | Val     |
|------------|---------|---------|
| Toxic      |  115,216 | 29,118  |
| Non-Toxic  |  1,009,610 | 225,154 |

Given the dataset's size exceeding 50MB, we opted to store it on Google Drive for convenient access and sharing. <ins>You must use your USC account to access</ins>:

- https://drive.google.com/drive/folders/1TeUC1swuEIScsIzI2Cnl28Yg3fF8vfzi?usp=sharing

# Evaluation
The datasets used for evalution were used as is an can be downloaded from the folder: `./eval` or the Google Drive (<ins>You must use your USC account to access</ins>): https://drive.google.com/drive/folders/1gr8sNuVJGfRygbsw0p9LjlTihuTBqkRb?usp=sharing

### <ins> DynaBench </ins>

DynaBench is an adversarially collected set of hate speech where human annotators create examples that an iteratively improved hate-speech classifier cannot detect.

### <ins> Microagressions.com</ins>

Is a publicly available Tumblr blog where users can anonymously post about socially biased interactions and utterances in the wild.

Contains data from [microaggressions.com](microaggressions.com).

### <ins> Social Bias Frames </ins>
SBF is a corpus of socially biased and offensive content from various online sources.