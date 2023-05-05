"""
Generate 1-100 random numbers directly from a logit file.
Namely, this is a valid approach since the prompt is fixed when we generate natively from a LLM.
"""

import numpy as np
import os
import pickle
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM


def sample_from_logits(vocab, logit_file, sample_size=1000):
    """
    Sample 1-100 token distribution from logits.
    Discard invalid generations
    """

    # Tuple of logits, where logits[i] is the RNG distribution for the (i + 1)-th generated token
    with open(logit_file, 'rb') as file:
        logits = pickle.load(file)
    # Since the 1st generated token is always whitespace, we look at the 2nd token distribution
    rng_logits = logits[0].numpy()[0]
    # Stable softmax
    z = rng_logits - max(rng_logits)
    rng_dist = np.exp(z) / np.sum(np.exp(z)) 
    print("rng_dist", rng_dist.size)
    print("vocab length", len(vocab))
    input()

    i = 0
    sample = []
    print("Vocab?", list(vocab.keys()))
    with open('alpaca_vocab.txt', 'w') as f:
        for item in list(vocab.keys()):
            f.write("%s\n" % item)

    while i < sample_size: 
        # raw_generation = str(np.random.choice(rng_dist))
        sampled_item = np.random.choice(list(vocab.keys()), p=rng_dist)
        print("sampled_item", sampled_item)
        # .read().decode('utf-8')
        try:
            token = float(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", sampled_item)[0])
            print("OMG")
            break
            if token in set(range(1, 100)):
                sample.append(token)
                i += 1
        except Exception as e:
            # print(f"Exception while sampling: {e}")
            continue
        
    return sample


# def iter_dir(dir_name='logits'):
    
#     for filename in os.listdir(dir_name):
#         # These are actually pickle files
#         if filename.endswith('.pt'):
#             file_path = os.path.join(dir_name, filename)
#             if 'alpaca' in filename and 'unmarked' in filename:
#                 alpaca_unmarked_fp = file_path
#             if 'flan' in filename and 'unmarked' in filename:
#                 flan_unmarked_fp = file_path
            
#     # Actually do tests
#     alpaca_ks = {}
#     flan_ks = {}

#     # Fix this later LOL, testing since we don't have unmarked RNG files for now
#     alpaca_unmarked_fp = 'rng/alpaca_marked_g10_d1_rep_10_4-28-17-36.npy'
#     flan_unmarked_fp = 'rng/flan_marked_g10_d1_rep_10_4-28-18-2.npy'

#     for filename in os.listdir(dir_name):
#         if filename.endswith('.pt'):
#             file_path = os.path.join(dir_name, filename)
#             if 'alpaca' in filename:
#                 stat, p_val = ks_test(alpaca_unmarked_fp, file_path)
#                 alpaca_ks[filename] = p_val
#             if 'flan' in filename:
#                 stat, p_val = ks_test(flan_unmarked_fp, file_path)
#                 flan_ks[filename] = p_val

    
    # with open('alpaca_ks.json', "w") as outfile:
    #     json.dump(alpaca_ks, outfile, indent=4)
    
    # with open('flan_ks.json', "w") as outfile:
    #     json.dump(flan_ks, outfile, indent=4)

def main():

    with open('alpaca_vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    print("Loaded Alpaca vocab!")
    # sample = np.array(sample_from_logits(tokenizer.get_vocab(), 'logits/alpaca_logits_unmarked.pt'))
    sample = np.array(sample_from_logits(vocab, 'logits/alpaca_logits_marked_g75_d100.pt'))
    np.save('rng/alpaca_unmarked.npy', sample)    
    
    # tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xxl')
    # vocab = tokenizer.get_vocab()
    # with open('alpaca_vocab.pkl', 'wb') as f:
    #     pickle.dump(vocab, f)
    # print("Loaded Flan tokenizer!")
    # sample = np.array(sample_from_logits(tokenizer.get_vocab(), 'logits/flan_logits_unmarked.pt'))

if __name__ == "__main__":
    main()
    
