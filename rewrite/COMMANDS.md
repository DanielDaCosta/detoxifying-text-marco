# Commmands used for testing


## Testing Masking and Generation
```
python -m rewrite.masking_v2

python -m rewrite.generation_v2
```
##  Rewriting

```
python -m rewrite.rewrite_example --data_path /Users/danieldacosta/Documents/USC/csci662/Project/detoxifying-text-marco/datasets/microagressions/val.csv --antiexpert_path /Users/danieldacosta/Documents/USC/csci662/Project/CARC_outputs/bart-base_1e-06_0_32_jigsaw_full_30/checkpoint-35000 --output_dir data/dexp_outputs --rep_penalty 1.0 --thresh 1.2 --temperature 2.5 --alpha_a 1.5 --alpha_e 4.25 --alpha_b 1.0 --filter_p 1.0 --batch_size 50 --max_length 128 --top_k 0 --top_p 1
```

