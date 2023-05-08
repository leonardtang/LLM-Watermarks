import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pickle
from matplotlib.widgets import Slider

FOLDER = "logits-pile-alpaca"


def average_logits(folder):
    accumulators = np.zeros((140,32000), dtype="float64")
    num_files = 0
    for i, name in enumerate(glob.glob(folder + "/*.pt")):
        with open(name, 'rb') as file:
            logits = pickle.load(file)
            if i == 0:
                accumulators[i] = logits[0].numpy()[0]
            else:
                accumulators[i] = accumulators[i - 1] + logits[0].numpy()[0]
    return accumulators

ACCUMULATORS = average_logits(FOLDER)

def construct_lorenz(gamma, delta, to_average):
    logits = ACCUMULATORS[int(to_average) - 1].copy()
    logits /= to_average
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
    to_average = averaging_slider.val
    lorenz = construct_lorenz(gamma, delta, to_average)
    lorenz = lorenz[~np.isinf(lorenz)]
    ax.clear()
    ax.hist(lorenz, bins=200, color='darkgreen', alpha=0.7)
    ax.set_xlabel('Cumulative Probability')
    ax.set_ylabel('Frequency')

    fig.canvas.draw_idle()

if __name__ == "__main__":
    lorenz = construct_lorenz(0.5, 10, 25)

    fig, ax = plt.subplots(figsize=[6, 6])
    plt.subplots_adjust(left=0.25, bottom=0.25)

    ax.hist(lorenz, bins=200, color='darkgreen', alpha=0.7)
    ax.set_xlabel('Cumulative Probability')
    ax.set_ylabel('Frequency')

    axcolor = 'lightgoldenrodyellow'
    ax_gamma = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_delta = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_average = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

    gamma_slider = Slider(ax_gamma, 'Gamma', 0, 1, valinit=0.5, valstep=0.01)
    delta_slider = Slider(ax_delta, 'Delta', 0, 100, valinit=10, valstep=1)
    averaging_slider = Slider(ax_average, 'Averaging', 1, 140, valinit=25, valstep=1)

    gamma_slider.on_changed(update)
    delta_slider.on_changed(update)
    averaging_slider.on_changed(update)

    plt.show()
