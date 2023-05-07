import matplotlib.pyplot as plt
import numpy as np
import pickle

def load_logits(file_name):
    with open(file_name, 'rb') as file:
        logits = pickle.load(file)
        return logits[0].numpy()[0]

def plot_logits(file_name1, file_name2):
    logits1 = load_logits(file_name1)
    logits2 = load_logits(file_name2)
    differences = logits1 - logits2

    fig, ax = plt.subplots(figsize=[6, 6])
    
    ax.scatter(np.arange(differences.size) / (differences.size - 1), differences, marker='o', color='red', s=1.5, label=file_name1)
    
    ax.set_xlabel('Token Ranking')
    ax.set_ylabel('Cumulative Probability')
    ax.legend()
    
    plt.show()

file_name1 = "./logits-algo-alpaca/alpaca_logits_unmarked_v_0.pt"
file_name2 = "./logits-algo-alpaca/alpaca_logits_unmarked_v_1.pt"

plot_logits(file_name1, file_name2)
