"""
Implementation of Watermarking algorithms from Kirchenbaur et. al (2023)
- Are these general? Like not specific to any model
"""

import locale
import matplotlib.pyplot as plt
import math
import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
gpt2large = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device)
vocab_size = gpt2.config.vocab_size

def sample(model, prompt, length, masker=None):
    """
    - Sample `length` tokens of output text from model given input `text`
    - `masker` is the watermark
    """

    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(length):
            
            logits = model(input_ids)
            if masker:
                # For now, only assume that watermark has access to prompt tokens
                mask = masker(input_ids) 
            else:
                mask = torch.zeros_like(logits[0][0][-1])
            
            # Mask bumps up certain tokens' probabilities in the vocabulary
            logits[0][0][-1] += mask
            probs = torch.softmax(logits[0][0][-1], dim=0)
            
            # TODO(ltang): move beyond multinomial decode to beam search
            sample = torch.multinomial(probs, 1)
            # Input window extends to include generated token
            # Affects masker/watermark via `hash_tensor``
            input_ids = torch.cat([input_ids, sample.unsqueeze(0)], dim=1)

    return input_ids, logits


def hash_tensor(tensor):
    return hash(tuple(tensor.tolist()))


def soft_watermark(input_ids, gamma, delta):
    """
    - Randomly partition vocabulary into green and red list. Bump up likelihood of green list tokens.
    - Gamma controls the green list size (specifically, what proportion of entire list is green)
    - Delta is the logit bump that gets added to tokens in the green list
    """

    green_list_length = int(vocab_size * gamma)
    # Use hash of token values to seed RNG
    random.seed(hash_tensor(input_ids[0]))
    # Use RNG to re-partition
    indices_to_mask = random.sample(range(vocab_size), green_list_length)
    mask = torch.zeros(vocab_size).to(device)
    mask[indices_to_mask] = delta  
    return mask


# TODO(ltang): should we be removing the prompt from text? Right now it seems like we might overcount green_count
def detect_soft_watermark(text, gamma):
    """
    - Given (human or machine generated) `text`, a combination of prompt + output, determine if it has been watermarked
    - Assume detector has access to hash function and RNG
    - Gamma is the green list proportion
    """

    tokens = tokenizer.encode(text)
    T = len(tokens)
    # Count the number of tokens in the test text that are green list tokens
    green_count = 0

    for i, token in enumerate(tokens):
        prev_tokens = tokens[:i]
        # Detector has access to hash function and RNG
        random.seed(hash(tuple(prev_tokens)))
        green_list_length = int(vocab_size * gamma)
        # Recover green list at each sample step
        green_list = set(random.sample(range(vocab_size), green_list_length))

        # Check if the current token is in the green list
        if token in green_list:
            green_count += 1

    # One proportion z-test to evaluate the null hypothesis (the text sequence is generated with no knowledge of the red list rule)
    z = 2 * (green_count - T / 2) / math.sqrt(T)
    return z


gamma = 0.5
watermarked = tokenizer.decode(sample(gpt2large, "The quick brown", 100, masker = lambda x: soft_watermark(x, gamma=gamma, delta=999))[0][0])
unmarked = tokenizer.decode(sample(gpt2large, "The quick brown", 100)[0][0])

print("WATERMARKED TEXT:")
print(watermarked)
print("UNMARKED TEXT:")
print(unmarked)

print("Z-Score on Watermarked Text")
print(detect_soft_watermark(watermarked, gamma))
print("Z-Score on Unmarked Text")
print(detect_soft_watermark(unmarked, gamma))

