"""
Two-sample Kolmogorov-Smirnov test between 1-100 RNG distribution of watermarked vs. unmarked LLMs
"""

import json
import numpy as np
import os
from scipy import stats

# TODO(ltang): test these with repeated distributions instead
def ks_test(unmarked_file, watermarked_file):
    
    unmarked_dist = np.concatenate(np.load(unmarked_file))
    watermarked_dist = np.concatenate(np.load(watermarked_file))

    assert len(unmarked_dist) == len(watermarked_dist)
    print(f"Unmarked File: {unmarked_file}")
    print(f"Watermarked File: {watermarked_file}")
    print(f"Total Sample Size: {len(unmarked_dist)}")

    # Hope this can just take in np.arrays
    test_res = stats.kstest(unmarked_dist, watermarked_dist)
    return test_res.statistic, test_res.pvalue


def iter_dir(dir_name='rng'):
    
    # TODO(ltang): clean up this hacky ass code
    for filename in os.listdir(dir_name):
        if filename.endswith('.npy'):
            file_path = os.path.join(dir_name, filename)
            if 'alpaca' in filename and 'unmarked' in filename:
                alpaca_unmarked_fp = file_path
            if 'flan' in filename and 'unmarked' in filename:
                flan_unmarked_fp = file_path
            
    # Actually do tests
    alpaca_ks = {}
    flan_ks = {}

    # Fix this later LOL, testing since we don't have unmarked RNG files for now
    alpaca_unmarked_fp = 'rng/alpaca_marked_g10_d1_rep_10_4-28-17-36.npy'
    flan_unmarked_fp = 'rng/flan_marked_g10_d1_rep_10_4-28-18-2.npy'

    for filename in os.listdir(dir_name):
        if filename.endswith('.npy'):
            file_path = os.path.join(dir_name, filename)
            if 'alpaca' in filename:
                stat, p_val = ks_test(alpaca_unmarked_fp, file_path)
                alpaca_ks[filename] = p_val
            if 'flan' in filename:
                stat, p_val = ks_test(flan_unmarked_fp, file_path)
                flan_ks[filename] = p_val

    
    with open('alpaca_ks.json', "w") as outfile:
        json.dump(alpaca_ks, outfile, indent=4)
    
    with open('flan_ks.json', "w") as outfile:
        json.dump(flan_ks, outfile, indent=4)


if __name__ == "__main__":
    iter_dir()
