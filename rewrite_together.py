import argparse
from pathlib import Path
from typing import Union, List
import os
from IPython import embed
from utils import *
import nltk.tokenize.casual
import sys
import pandas as pd
from tqdm import tqdm
import re
import time
import itertools
import sentencepiece

import requests
import together

"""
Method to read in data from one of the datasets we ran experiments on
Additional code needed if you want to add your own datasets here
- Returns a list of texts to rewrite
"""
def get_data(args):
    # inputs =  ["I'm surprised you got it done, seeing as you're all girls!", "You are a human", "You are a genius"]
    inputs =  []
    if args.data_path is not None:
        if "dynabench" in args.data_path:
            # DATA PATHS:
                # VALIDATION: "../datasets/dynabench/db_dev.csv"
                # TEST: "../datasets/dynabench/db_test.csv"
            df = pd.read_csv(args.data_path)
            df_lab = "hate"
            if "dev" in args.data_path:
                df_split = "dev"
            elif "test" in args.data_path:
                df_split = "test"
            else:
                raise Exception('Reproduciton study: should only run rewrites for dev and test sets')

            # if "test" in args.data_type:
            #     df_split = "test"
            # if "train" in args.data_type:
            #     df_split = "train"
            # if "nothate" in args.data_type:
            #     df_lab = "nothate"
            # if "all" in args.data_type:
            #     inputs = df[(df.split == df_split) & (df.label == df_lab)].text.tolist()
            # else:
            #     df_round = int(args.data_type[-1])
            #     inputs = df[(df.split == df_split) & (df.label == df_lab)][df["round.base"] == df_round].text.tolist()
            inputs = df[(df.split == df_split) & (df.label == df_lab)].text.tolist()
            
        elif "sbf" in args.data_path:
            # DATA PATHS:
                # VALIDATION: "../datasets/sbf/sbfdev.csv"
                # TEST: "../datasets/sbf/sbftst.csv"
            df = pd.read_csv(args.data_path)
            dataSource = "redditMicroagressions"
            
            # if "nonoff" in args.data_type:  # non-offensive
            #     inputs = df[df.dataSource ==dataSource ][df.offensiveYN < 0.5].post.tolist()
            # else:
            #     inputs = df[df.dataSource ==dataSource ][df.offensiveYN >= 0.5].post.tolist()
            inputs = df[df.dataSource ==dataSource ][df.offensiveYN >= 0.5].post.tolist()

        elif "microagressions" in args.data_path:
            # DATA PATHS:
                # VALIDATION: "../datasets/microagressions/val.csv"
                # TEST: "../datasets/microagressions/test.csv"
            df = pd.read_csv(args.data_path)
            inputs = [preprocess(s) for s in df.actual_quote.tolist()]

    return inputs

def get_prompt_llama(inp):
    return f'<s>[INST] <<SYS>>\nYou are a concise assistant helping to remove toxicity from text by replacing toxic terms.\nYour goal is to reduce toxicity while changing the meaning and structure of the original text as little as possible. Respond to user requests with only the rephrased text and without additional language.\n<</SYS>>\n\nRephrase the following text to make it non-toxic: "{inp}" [/INST]'

def get_prompt(inp):
    sys_msg = '<<SYS>>\nYou are a concise assistant helping to remove toxicity from text by replacing toxic terms.\nYour goal is to reduce toxicity while changing the meaning and structure of the original text as little as possible. Respond to user requests with only the rephrased text and without additional language.\n<</SYS>>\n\n'
    inst_msg = f'[INST] Rephrase the following text to make it non-toxic: "{inp}" [/INST]'
    return ''.join([sys_msg, inst_msg])

def get_prompt_mistral(inp):
    return f'<s>[INST] Reduce toxicity by replacing toxic terms while changing the meaning and structure of the original text as little as possible. Do not add additional content to the text or embelish it. Only replace toxic terms. Respond with only the rephrased text and without additional language.\nRephrase the following text to make it non-toxic: "{inp}" [/INST]'

def est_budget(inputs, args):
    # Assuming use of llama models on together.ai
    pptt = None
    if args.model == 'togethercomputer/llama-7b-chat':
        pptt = 0.0002
    elif args.model == 'togethercomputer/llama-2-13b-chat':
        pptt = 0.000225
    elif args.model == 'togethercomputer/llama-2-70b-chat':
        pptt = 0.0009

    elif args.model == 'mistralai/Mistral-7B-Instruct-v0.1':
        pptt = 0.0002
    elif args.model == 'NousResearch/Nous-Hermes-llama-2-7b':
        pptt = 0.0002

    print(f"\nAssuming price per 1k tokens = {pptt}")

    sp = sentencepiece.SentencePieceProcessor(model_file='rewrite/tokenizer.model')
    total_tok = sum([len(sp.encode_as_ids(' '.join([get_prompt(i), i]))) for i in inputs])
    print(f'Estimated total tokens in/out: {total_tok}')

    estimate = pptt * total_tok / 1000
    print(f"Estimated cost of rewrites: ${estimate}")

    proceed = input('Budget OK? enter "GO" to proceed.\n')
    if proceed != "GO":
        print('Rewrites canceled, Exiting.')
        exit()
    print()

def do_rewrites(inputs, args):
    if 'llama' in args.model:
        prompt_generator = get_prompt_llama
    elif 'mistral' in args.model:
        prompt_generator = get_prompt_mistral

    outputs = []
    for i in inputs:
        rephrase = together.Complete.create(
            prompt = prompt_generator(i), 
            model = args.model, 
            max_tokens = 128,
            temperature = 0,
            top_k = 1,
            top_p = 1,
            repetition_penalty = 1,
            stop = ['</s>']
        )
        outputs.append(rephrase['output']['choices'][0]['text'].strip('\n').strip('"'))
        # print(rephrase)
        time.sleep(1.1)

    return outputs

def rewrite(args):
    # Get the inputs to rewrite
    inputs = get_data(args)

    if args.check_cost:
        est_budget(inputs, args)

    outputs = do_rewrites(inputs, args)

    # Save the original inputs and the generations
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_name = args.model.replace('/', '_')
    with open(os.path.join(args.output_dir, f"{model_name}-orig.txt"), "w") as f:
        for l in inputs:
            f.write(re.sub(r"\s+", " ", l).strip() + "\n")
    with open(os.path.join(args.output_dir, f"{model_name}-gen.txt"), "w") as f:
        for l in outputs:
            f.write(re.sub(r"\s+", " ", l).strip() + "\n")

    print(time.strftime('%l:%M%p %Z on %b %d, %Y').strip())
    print("Finished generation\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General args
    parser.add_argument("--data_path", default =None, help = "Path to the data")
    parser.add_argument("--output_dir", default = None, help = "Path to save the outputs to")

    parser.add_argument('--check_cost', action='store_true', help = "Check budget before execution")
    parser.add_argument('--no_check_cost', dest='check_cost', action='store_false', help = "Don't check budget before execution")
    parser.set_defaults(check_cost=True)

    parser.add_argument("--api_url", default ="https://api.together.xyz/inference", help = "Path to the data")
    parser.add_argument("--model", default ="mistralai/Mistral-7B-Instruct-v0.1", help = "Path to the data")
    # [togethercomputer/llama-7b-chat, togethercomputer/llama-2-13b-chat, togethercomputer/llama-2-70b-chat]
    parser.add_argument("--api_key", default =None, help = "Path to the data")

    args = parser.parse_args()

    if not args.api_key:
        from private import API_KEY
        args.api_key = API_KEY
    together.api_key = args.api_key

    rewrite(args)