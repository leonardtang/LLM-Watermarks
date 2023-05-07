import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pickle
from matplotlib.widgets import Slider

FOLDER = "logits-algo-alpaca"

def average_logits(folder):
    accumulator = np.zeros((32000,), dtype="float64")
    for name in glob.glob(folder + "/*.pt"):
        with open(name, 'rb') as file:
            print("Loaded a file!")
            logits = pickle.load(file)[0].numpy()[0]
            print(logits)
            accumulator += logits
    accumulator *= 0.01
    return accumulator

ACCUMULATED = average_logits(FOLDER)

def construct_lorenz(gamma, delta):
    logits = ACCUMULATED.copy()
    len_greenlist = int(len(logits) * gamma)
    greenlist = len_greenlist * [1] + (len(logits) - len_greenlist) * [0]
    watermarks = np.array(greenlist)
    rng = np.random.default_rng(0)
    rng.shuffle(watermarks)
    logits += watermarks * delta
    return logits

def update(val):
    gamma = gamma_slider.val
    delta = delta_slider.val
    lorenz = construct_lorenz(gamma, delta)

    ax.clear()
    ax.plot([0, 1], [0, 1], color='k', linestyle='dashed', linewidth=1)
    ax.scatter(np.arange(lorenz.size) / (lorenz.size - 1), lorenz, marker='o', color='darkgreen', s=1.5)
    ax.set_xlabel('Token Ranking')
    ax.set_ylabel('Cumulative Probability')

    fig.canvas.draw_idle()

lorenz = construct_lorenz(0.5, 10)

fig, ax = plt.subplots(figsize=[6, 6])
plt.subplots_adjust(left=0.25, bottom=0.25)

ax.scatter(np.arange(lorenz.size) / (lorenz.size - 1), lorenz, marker='o', color='darkgreen', s=1.5)
ax.set_xlabel('Token Ranking')
ax.set_ylabel('Cumulative Probability')

axcolor = 'lightgoldenrodyellow'
ax_gamma = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_delta = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

gamma_slider = Slider(ax_gamma, 'Gamma', 0, 1, valinit=0.5, valstep=0.01)
delta_slider = Slider(ax_delta, 'Delta', 0, 100, valinit=10, valstep=1)

gamma_slider.on_changed(update)
delta_slider.on_changed(update)

plt.show()
