"""
Generate 1-100 random numbers directly from a logit file.
Namely, this is a valid approach since the prompt is fixed when we generate natively from a LLM.
"""

import argparse
import numpy as np
import os
import pickle
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM


def stable_softmax(logits):
    """
    Numerically stable softmax
    """
    z = logits - max(logits)
    dist = np.exp(z) / np.sum(np.exp(z)) 
    return dist


def sample_from_logits(tokenizer, logit_file, sample_size=1000):
    """
    Sample 1-100 token distribution from logits.
    Discard invalid generations
    """

    # Tuple of logits, where logits[i] is the RNG distribution for the (i + 1)-th generated token
    with open(logit_file, 'rb') as file:
        logits = pickle.load(file)
    # Since the 1st generated token is always whitespace, we look at the 2nd token distribution
    rng_logits = logits[1].numpy()[0]
    rng_dist = stable_softmax(rng_logits)

    i = 0
    sample = []
    n = len(rng_dist)
    while i < sample_size:
        if i % (sample_size // 5) == 0: print(f'On sample {i} of {sample_size}')
        sampled_idx = np.random.choice(n, p=rng_dist)
        sampled_word = tokenizer.convert_ids_to_tokens(sampled_idx)
        try:
            token = float(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", sampled_word)[0])
            if token in set(range(1, 101)):
                sample.append(token)
                i += 1
        except Exception as e:
            continue
        
    return sample


def main(model, logit_file, out_file):

    if model == 'alpaca-lora':
        tokenizer = LlamaTokenizer.from_pretrained('chainyo/alpaca-lora-7b')
        print("Loaded Alpaca tokenizer!")    

    elif model == 'flan-t5':
        tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xxl')
        print("Loaded Flan tokenizer!")

    sample = np.array(sample_from_logits(tokenizer, logit_file))
    np.save(out_file, sample)


# Example calls: 
# python logit_rng.py -m alpaca-lora -lf logits-v4/alpaca_logits_unmarked.pt -of rng-logits/alpaca_unmarked.npy
# python logit_rng.py -m flan-t5  -lf logits-v4/flan_logits_unmarked.pt -of rng-logits/flan_unmarked.npy
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--logit-file', '-lf', type=str)
    parser.add_argument('--out-file', '-of', type=str)
    args = parser.parse_args()
    main(args.model, args.logit_file, args.out_file)
