"""
Construct ranked token Lorenz curves from logits files and perform Gini/smoothness/spiking analysis on them.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
import pickle
import re


def construct_lorenz(logit_file):

    with open(logit_file, 'rb') as file:
        logits = pickle.load(file)

    # TODO(ltang): investigate which index to select logits from
    logits = logits[1].numpy()[0]

    # TODO(ltang): figure out how to deal with probability vs. logits
    z = logits - max(logits)
    probs = np.exp(z) / np.sum(np.exp(z)) 
    
    # TODO(ltang): normalization schemes to consider...?
    # logits = (logits - logits.min()) / (logits.max() - logits.min())
    # logits = logits / np.linalg.norm(logits)

    sorted_probs = probs.copy()
    sorted_probs.sort()
    
    lorenz = sorted_probs.cumsum() / sorted_probs.sum()
    lorenz = np.insert(lorenz, 0, 0)

    match_gamma = re.search(r'g(\d+)_', logit_file)
    match_delta = re.search(r'd(\d+).', logit_file)
    match_model = re.search(r'^([^_]+)', logit_file)

    if match_model:
        model_name = match_model.group(1).split('/')[-1]
    else:
        raise Exception('No model name found in logit file')

    # Lorenz Curve  
    fig, ax = plt.subplots(figsize=[6,6])
    ax.scatter(np.arange(lorenz.size) / (lorenz.size - 1), lorenz, marker='o', color='darkgreen', s=1.5)
    # Equality Line
    ax.plot([0,1], [0,1], color='k', linestyle='dashed', linewidth=1)
    ax.set_xlabel('Token Ranking')
    ax.set_ylabel('Cumulative Probability')
    
    # Gini 
    g_coeff = gini(sorted_probs)
    # Max Diff
    m_diff = max_diff(sorted_probs)
    # Avg Diff
    avg_diff = average_diff(sorted_probs)

    if match_gamma and match_delta:
        ax.set_title(f'{model_name.title()} Lorenz Curve for Gamma {int(match_gamma.group(1)) / 100} and Delta {match_delta.group(1)}')
        plt.savefig(f'lorenz_plots/{model_name}_g{match_gamma.group(1)}_d{match_delta.group(1)}.png')
        print(f"{model_name.title()} Gini at gamma {match_gamma.group(1)} and delta {match_delta.group(1)}: {g_coeff}")
        print(f"{model_name.title()} Max diff at gamma {match_gamma.group(1)} and delta {match_delta.group(1)}: {m_diff}")
        return g_coeff, m_diff, avg_diff, int(match_gamma.group(1)), int(match_delta.group(1)), model_name
    else:
        ax.set_title(f'{model_name.title()} Lorenz Curve for Unmarked Model')
        plt.savefig(f'lorenz_plots/{model_name}_unmarked.png')
        print(f"{model_name.title()} Gini unmarked: {g_coeff}")
        print(f"{model_name.title()} Max diff unmarked: {m_diff}")
        return g_coeff, m_diff, avg_diff, 0, 0, model_name
    

def gini(sorted_arr):
    """
    Compute Gini coefficient of (normalized) logits or probability distribution
    """
    n = sorted_arr.size
    coef_ = 2. / n
    const_ = (n + 1.) / n
    weighted_sum = sum([(i + 1) * y_i for i, y_i in enumerate(sorted_arr)])
    return coef_ * weighted_sum / (sorted_arr.sum()) - const_


def max_diff(a):
    """
    Maximum difference between consecutive elements in a list
    """
    return max(np.diff(a))


def average_diff(a):
    """
    Average difference between consecutive elements in a list
    """
    return np.mean(np.diff(a))


def remap_keys(mapping):
    """
    For dumping tuples and floats in JSON
    """
    return [{'key': k, 'value': float(v)} for k, v in mapping.items()]


def main(dir_name='logits'):

    ## Various Detection Metrics
    gini_metrics = {}
    m_diff_metrics = {}
    a_diff_metrics = {}
    
    for filename in sorted(os.listdir(dir_name)):
        if filename.endswith('.pt'):
            file_path = os.path.join(dir_name, filename)
            g_c, m_d, a_d, gamma, delta, model = construct_lorenz(file_path)
            gini_metrics[gamma, delta, model] = g_c
            m_diff_metrics[gamma, delta, model] = m_d
            a_diff_metrics[gamma, delta, model] = a_d

    gini_metrics, m_diff_metrics, a_diff_metrics = dict(sorted(gini_metrics.items())), dict(sorted(m_diff_metrics.items())), dict(sorted(a_diff_metrics.items()))    

    with open('gini.json', 'w') as f:
        json.dump(remap_keys(gini_metrics), f, indent=4)

    with open('max_diff.json', 'w') as f:
        json.dump(remap_keys(m_diff_metrics), f, indent=4)

    with open('avg_diff.json', 'w') as f:
        json.dump(remap_keys(a_diff_metrics), f, indent=4)

if __name__ == "__main__":
    main()