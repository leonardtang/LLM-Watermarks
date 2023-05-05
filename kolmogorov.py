"""
Two-sample Kolmogorov-Smirnov test between 1-100 RNG distribution of watermarked vs. unmarked LLMs
"""

import json
import numpy as np
import os
from scipy import stats

# TODO(ltang): test these with repeated distributions instead
def ks_test(unmarked_file, watermarked_file):
    
    print(f"Unmarked File: {unmarked_file}")
    print(f"Watermarked File: {watermarked_file}")
    
    ## Treat as one big distribution
    # unmarked_dist = np.concatenate(np.load(unmarked_file))
    # watermarked_dist = np.concatenate(np.load(watermarked_file))
    # assert len(unmarked_dist) == len(watermarked_dist)
    # print(f"Total Sample Size: {len(unmarked_dist)}")

    ## Hope this can just take in np.arrays
    # test_res = stats.kstest(unmarked_dist, watermarked_dist)
    # return test_res.statistic, test_res.pvalue

    # Average across samples
    avg_stat, avg_p = 0, 0
    unmarked_dists = np.load(unmarked_file)
    watermarked_dists = np.load(watermarked_file)

    for u in unmarked_dists:
        for w in watermarked_dists:
            test_res = stats.kstest(u, w)
            avg_stat += test_res.statistic
            avg_p += test_res.pvalue

    total_pw = len(unmarked_dists) * len(watermarked_dists)
    return avg_stat / total_pw, avg_p / total_pw


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

    for filename in os.listdir(dir_name):
        if filename.endswith('.npy'):
            file_path = os.path.join(dir_name, filename)
            if 'alpaca' in filename:
                stat, p_val = ks_test(alpaca_unmarked_fp, file_path)
                alpaca_ks[filename] = p_val
            if 'flan' in filename:
                stat, p_val = ks_test(flan_unmarked_fp, file_path)
                flan_ks[filename] = p_val

    
    alpaca_ks, flan_ks = dict(sorted(alpaca_ks.items())), dict(sorted(flan_ks.items()))

    with open('alpaca_ks.json', "w") as outfile:
        json.dump(alpaca_ks, outfile, indent=4)
    
    with open('flan_ks.json', "w") as outfile:
        json.dump(flan_ks, outfile, indent=4)


if __name__ == "__main__":
    iter_dir()
