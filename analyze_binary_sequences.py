import openai
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import math
from api_keys import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

# Generate a string of 0s and 1s using GPT-3 and return token probabilities
def generate_binary_string_and_probabilities(prompt, length=1, get_probs=' 0'):
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=length,
        temperature=0.7,
        logprobs=2,
    )

    binary_string = response.choices[0].text.strip()
    token_logprobs = response.choices[0].logprobs.top_logprobs
    probabilities = [
        {token: math.exp(logprob) for token, logprob in token_probs.items()}
        for token_probs in token_logprobs
    ]
    probs = [token_probs.get(get_probs, 0) for token_probs in probabilities]
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
def generate_demo_sequence_probabilities(iterations, demo_sequence_length, digits=[0, 1]):
    zero_probs = []
    for _ in range(iterations):
        random_demo, digs = random_demo_sequence(demo_sequence_length, digits)
        prompt = f"Choose two digits, and generate a uniformly random string of those digits. Previous digits should have no influence on future digits: {random_demo}"
        zero_prob = generate_binary_string_and_probabilities(prompt, length=1, get_probs=' ' + str(digs[0]))[1]
        zero_probs.extend(zero_prob)
    return zero_probs

# Plot the token probabilities as a histogram
def plot_token_probabilities_histogram(zero_probs, bins=100, save_folder=None):
    plt.hist(zero_probs, label='0', bins=bins, alpha=0.7)
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Token Probabilities Histogram')
    plt.legend()
    if save_folder is not None:
        plt.savefig(save_folder + str(time.time()) + '.png')
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

# Generate probabilities using random demo sequences
probabilities = generate_demo_sequence_probabilities(iterations=100, demo_sequence_length=1000, digits=[0,1])
print(probabilities)
confidence_level = 0.95
lower_bound, upper_bound = hoefding_confidence_interval(probabilities, confidence_level)
print(f"95% confidence interval for the probability of 0: [{lower_bound:.4f}, {upper_bound:.4f}]")
plot_token_probabilities_histogram(probabilities, save_folder='plots/')