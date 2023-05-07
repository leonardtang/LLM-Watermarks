import openai
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import math
import torch
from scipy.stats import shapiro, kurtosis
from api_keys import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

# Generate a string of 0s and 1s using GPT-3 and return token probabilities
def generate_binary_string_and_probabilities(prompt, length=1, engine="ada", zero_token=' 0', one_token=' 1'):
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=length,
        temperature=0.7,
        logprobs=5,
    )

    binary_string = response.choices[0].text.strip()
    token_logprobs = response.choices[0].logprobs.top_logprobs
    probabilities = [
        {token: math.exp(logprob) for token, logprob in token_probs.items()}
        for token_probs in token_logprobs
    ]
    probs = [token_probs.get(zero_token, 0) / (token_probs.get(zero_token, 0) + token_probs.get(one_token, 0)) for token_probs in probabilities if token_probs.get(zero_token, 0) > 0 and token_probs.get(one_token, 0) > 0]
    return binary_string, probs

# Plot the token probabilities as a histogram
def plot_token_probabilities_histogram(zero_probs):

    plt.hist([zero_probs], label=['0'], bins=100, alpha=0.7)
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Token Probabilities Histogram')
    plt.legend()
    plt.show()

# Generate random binary demo sequence
def random_demo_sequence(length, digits=[0, 1]):
    if digits is None:
        digits = random.sample(range(10), 2)
    return ' '.join([str(random.choice(digits)) for _ in range(length)]), digits

# Generate demo sequence probabilities
def generate_demo_sequence_probabilities(iterations, demo_sequence_length, length=1, engine="ada", digits=[0, 1]):
    zero_probs = []
    for _ in range(iterations):
        random_demo, digs = random_demo_sequence(demo_sequence_length, digits)
        prompt = f"Choose two digits, and generate a uniformly random string of those digits. Previous digits should have no influence on future digits: {random_demo}"
        zero_prob = generate_binary_string_and_probabilities(prompt, length=length, engine=engine, zero_token=' ' + str(digs[0]))[1]
        zero_probs.extend(zero_prob)
    return zero_probs

def generate_demo_sequence_average_probabilities(iterations, n_average, demo_sequence_lengths, digits=[0, 1]):
    probs = []
    for i in range(iterations):
        random_demo_1, _ = random_demo_sequence(demo_sequence_lengths[1], digits)
        average_prob = 0
        for j in range(n_average):
            random_demo_0, _ = random_demo_sequence(demo_sequence_lengths[0], digits)
            prompt = f"Choose two digits, and generate a uniformly random string of those digits. Previous digits should have no influence on future digits: {random_demo_0} {random_demo_1}"
            to_add = generate_binary_string_and_probabilities(prompt, length=1, zero_token=' ' + str(digits[0]))
            # print(to_add, i, j)
            average_prob += to_add[1][0]/n_average
        probs.append(average_prob)
    return probs

# Save the token probabilities as .pt file
def save_token_probabilities(zero_probs, save_path=None):
    torch.save(zero_probs, save_path)

# Plot the token probabilities as a histogram
def plot_token_probabilities_histogram(zero_probs, bins=100, save_path=None, show=True):
    plt.hist(zero_probs, label='0', bins=bins, alpha=0.7)
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Token Probabilities Histogram')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

def hoefding_confidence_interval(probabilities, confidence_level=0.95):
    n = len(probabilities)
    empirical_mean = np.mean(probabilities)

    # Calculate Hoefding's bound (epsilon) using the formula
    # epsilon = sqrt((1/2n) * log(2/ (1 - confidence_level)))
    epsilon = np.sqrt((1 / (2 * n)) * np.log(2 / (1 - confidence_level)))

    # Calculate the lower and upper bounds
    lower_bound = max(0, empirical_mean - epsilon)
    upper_bound = min(1, empirical_mean + epsilon)

    return lower_bound, upper_bound

# Uses the Shapiro-Wilk test to determine if the data is normally distributed, and returns the p-value
def is_normal(probabilities, alpha=0.05):
    stat, p = shapiro(probabilities)
    kurt = kurtosis(probabilities)

    return p, p > alpha, {"s-w": stat, "kurt": kurt}

def logprobs_normal(probabilities, alpha=0.05):
    logprobs = np.log(probabilities)
    stat, p = shapiro(logprobs)
    kurt = kurtosis(logprobs)

    return p, p > alpha, {"s-w": stat, "kurt": kurt}

engine_options = ["ada", "babbage", "curie", "davinci", "text-ada-001", "text-babbage-001", "text-curie-001", "text-davinci-002", "text-davinci-003"]
for engine in engine_options:
    print(f"Engine: {engine}")
    iterations, demo_sequence_length, length = 50, 20, 100
    # Generate probabilities using random demo sequences
    # probabilities = generate_demo_sequence_probabilities(iterations=iterations, demo_sequence_length=demo_sequence_length, length=length, engine=engine, digits=[0,1])
    # probabilities = generate_demo_sequence_average_probabilities(iterations=10, n_average=10, demo_sequence_lengths=[15, 5], digits=[0,1])
    probabilities = torch.load('binary_data/Engine: ' + engine + ' Iterations: ' + str(iterations) + ' Demo Sequence Length: ' + str(demo_sequence_length) + ' Length: ' + str(length) + '.pt')
    print(f"Proportion of valid samples: {len(probabilities) / (iterations * length)}")
    confidence_level = 0.95
    lower_bound, upper_bound = hoefding_confidence_interval(probabilities, confidence_level)
    print(f"95% confidence interval for the probability of 0: [{lower_bound:.4f}, {upper_bound:.4f}]")
    p, normal, stats = is_normal(probabilities)
    print(f"Are the probabilities normally distributed? {is_normal}")
    print(stats.get("s-w"), stats.get("kurt"))
    print(f"Are the log probabilities normally distributed? {logprobs_normal(probabilities)}")
    print(f"Mean: {np.mean(probabilities):.4f}")
    print(f"Standard deviation: {np.std(probabilities):.4f}")
    save_token_probabilities(probabilities, save_path='binary_data/Engine: ' + engine + ' Iterations: ' + str(iterations) + ' Demo Sequence Length: ' + str(demo_sequence_length) + ' Length: ' + str(length) + '.pt')
    plot_token_probabilities_histogram(probabilities, show=False, save_path='plots/Engine: ' + engine + ' Iterations: ' + str(iterations) + ' Demo Sequence Length: ' + str(demo_sequence_length) + ' Length: ' + str(length) + '.png')
