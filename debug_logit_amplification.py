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
    
    fig, ax = plt.subplots(figsize=[6, 6])
    
    ax.scatter(np.arange(logits1.size) / (logits1.size - 1), logits1, marker='o', color='red', s=1.5, label=file_name1)
    ax.scatter(np.arange(logits2.size) / (logits2.size - 1), logits2, marker='o', color='blue', s=1.5, label=file_name2)
    
    ax.set_xlabel('Token Ranking')
    ax.set_ylabel('Cumulative Probability')
    ax.legend()
    
    plt.show()

file_name1 = "./logits-algo-alpaca/alpaca_logits_unmarked_v_0.pt"
file_name2 = "./logits-algo-alpaca/alpaca_logits_unmarked_v_1.pt"

plot_logits(file_name1, file_name2)
