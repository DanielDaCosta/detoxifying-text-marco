# Reproduction Study: MaRCo Detoxification
This is the repository for the 2023 ACL Paper ["Detoxifying Text with MaRCo: Controllable Revision with Experts and Anti-Experts"](https://arxiv.org/abs/2212.10543)

<p align="center">
  <img src="https://pbs.twimg.com/media/FkeZuLBUUAA6abX?format=jpg&name=4096x4096" alt="drawing" width="75%"/>
</p>

Author's original [Github Repo](https://github.com/shallinan1/MarcoDetoxification) 

## Dependencies

### <ins>Setting up the Environment</ins>
To set up the environment to run the code, make sure to have conda installed, then run

    conda env create -f environment.yml

Then, activate the environment

    conda activate rewrite

**Important!**: The environment is setup to run with a CUDA version compatible with RTX6000 GPUs. You may need to update the environment based on your own GPUs.

### <ins>Compute Requirements</ins>

We recommend using a single RTX6000 GPU (this is what we used for our experiments) or another NVIDIA GPU with >24GB VRAM to enable large-scale rewriting (large batch size). Our method can also run on smaller models <24GB VRAM, but you should set the batch size to be lower.

## Datasets and Preprocess
See `datasets/README.md` for access to the datasets and a description.

## Training
See `training/README.md` for code and commands.

The pre-trained models used were downloaded from hugging face:
- BART-Base: https://huggingface.co/facebook/bart-base
- BART-Small: https://huggingface.co/lucadiliello/bart-small
- Mistral-7B: [Together AI pre-trainned model](https://docs.together.ai/docs/inference-models)

All batch scripts utilized for jobs on CARC can be located in the `CARC/` folder, alongside the respective outputs and training logs for each job.


## Evaluation
See `evaluation/README.md` for code and commands.

<!-- ### <ins>Detoxification with MaRCo</ins>

See `rewrite/README.md` for details on how to run the detoxification pipeline, -->



## Citing this Work
If you use/reference this work, please cite us with:

    @inproceedings{hallinan-etal-2023-detoxifying,
        title = "Detoxifying Text with {M}a{RC}o: Controllable Revision with Experts and Anti-Experts",
        author = "Hallinan, Skyler  and
          Liu, Alisa  and
          Choi, Yejin  and
          Sap, Maarten",
        booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
        month = jul,
        year = "2023",
        address = "Toronto, Canada",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.acl-short.21",
        doi = "10.18653/v1/2023.acl-short.21",
        pages = "228--242",
        abstract = "Text detoxification has the potential to mitigate the harms of toxicity by rephrasing text to remove offensive meaning, but subtle toxicity remains challenging to tackle. We introduce MaRCo, a detoxification algorithm that combines controllable generation and text rewriting methods using a Product of Experts with autoencoder language models (LMs). MaRCo uses likelihoods under a non-toxic LM (expert) and a toxic LM (anti-expert) to find candidate words to mask and potentially replace. We evaluate our method on several subtle toxicity and microaggressions datasets, and show that it not only outperforms baselines on automatic metrics, but MaRCo{'}s rewrites are preferred 2.1 times more in human evaluation. Its applicability to instances of subtle toxicity is especially promising, demonstrating a path forward for addressing increasingly elusive online hate.",
    }


