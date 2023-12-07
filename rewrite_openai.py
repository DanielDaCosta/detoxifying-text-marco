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
from openai import OpenAI

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

def get_prompt(inp):
    return [
                {"role": "system", "content": "You are a concise assistant helping to remove toxicity from text by replacing toxic terms.\nYour goal is to reduce toxicity while changing the meaning and structure of the original text as little as possible. Respond to user requests with only the rephrased text and without additional language."},
                {"role": "user", "content": f'Rephrase the following text to make it non-toxic: "{inp}"'}
            ]

def est_budget(inputs, args):
    # Assuming use of llama models on together.ai
    pptt = None
    if args.model == 'gpt-3.5-turbo-1106':
        pptt = 0.0010
    elif args.model == 'gpt-4-0314':
        pptt = 0.03

    print(f"\nAssuming price per 1k tokens = {pptt}")

    sp = sentencepiece.SentencePieceProcessor(model_file='rewrite/tokenizer.model')
    get_prompt_str = lambda x: ' '.join([m['content'] for m in get_prompt(x)])
    total_tok = sum([len(sp.encode_as_ids(' '.join([get_prompt_str(i), i]))) for i in inputs])
    print(f'Estimated total tokens in/out: {total_tok}')

    estimate = pptt * total_tok / 1000
    print(f"Estimated cost of rewrites: ${estimate}")

    proceed = input('Budget OK? enter "GO" to proceed.\n')
    if proceed != "GO":
        print('Rewrites canceled, Exiting.')
        exit()
    print()

def do_rewrites(inputs, args):
    client = OpenAI()

    outputs = []
    for i in inputs:
        response = client.chat.completions.create(
            model=args.model,
            messages=get_prompt(i),
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=128,
            n=1
        )
        outputs.append(response.choices[0].message.content)
        time.sleep(0.2)  # I guess this'll work lol

        # TODO: TESTING
        if len(outputs) == 1:
            for i, o in zip(inputs[:5], outputs):
                print(f"SAMPLE:\n\tIN-PUT: {i}\n\t\OUTPUT: {o}")
            exit()
        # TODO: TESTING


    return outputs

def rewrite(args):
    # Get the inputs to rewrite
    inputs = get_data(args)

    est_budget(inputs, args)

    outputs = do_rewrites(inputs, args)

    # Save the original inputs and the generations
    os.makedirs(final_path, exist_ok=True)

    with open(os.path.join(final_path, f"{args.model}-orig.txt"), "w") as f:
        for l in inputs:
            f.write(re.sub(r"\s+", " ", l).strip() + "\n")
    with open(os.path.join(final_path, f"{args.model}-gen.txt"), "w") as f:
        for l in outputs:
            f.write(re.sub(r"\s+", " ", l).strip() + "\n")

    print(time.strftime('%l:%M%p %Z on %b %d, %Y').strip())
    print("Finished generation\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General args
    parser.add_argument("--data_path", default =None, help = "Path to the data")
    parser.add_argument("--output_path", default = None, help = "Path to save the outputs to")

    parser.add_argument("--model", default ="gpt-3.5-turbo-1106", help = "Path to the data")
    # [gpt-3.5-turbo-1106, gpt-4-0314]
    parser.add_argument("--api_key", default =None, help = "Path to the data")

    args = parser.parse_args()

    if not args.api_key:
        from private import OPENAI_API_KEY
        args.api_key = OPENAI_API_KEY
    os.environ['OPENAI_API_KEY'] = args.api_key

    rewrite(args)