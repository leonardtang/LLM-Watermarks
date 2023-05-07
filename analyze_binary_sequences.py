import openai
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import math
import torch
from scipy.stats import shapiro, kurtosis, skew
import argparse
import diptest
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

def test_bimodality(probabilities):
    skewness = skew(probabilities)
    kurt = kurtosis(probabilities)
    n = len(probabilities)
    bimodality_coeff = (skewness**2 + 1) / (kurt + 3*(n-1)**2/((n-2)*(n-3)))
    dip = diptest.dipstat(probabilities)
    return bimodality_coeff, dip

def main(engine, iterations, demo_sequence_length, length, load_probabilities, plot, tests):
    if load_probabilities:
        probabilities = torch.load(f'binary_data/Engine: {engine} Iterations: {iterations} Demo Sequence Length: {demo_sequence_length} Length: {length}.pt')
    else:
        # Generate probabilities using random demo sequences
        probabilities = generate_demo_sequence_probabilities(iterations=iterations, demo_sequence_length=demo_sequence_length, length=length, engine=engine, digits=[0,1])

    if "valid_proportion" in tests:
        print(f"Proportion of valid samples: {len(probabilities) / (iterations * length)}")

    if "confidence_interval" in tests:
        confidence_level = 0.95
        lower_bound, upper_bound = hoefding_confidence_interval(probabilities, confidence_level)
        print(f"95% confidence interval for the probability of 0: [{lower_bound:.4f}, {upper_bound:.4f}]")

    if "normality" in tests or "kurtosis" in tests or "shapiro_wilk" in tests:
        p, normal, stats = is_normal(probabilities)

    if "normality" in tests:
        print(f"Are the probabilities normally distributed? {normal}")

    if "shapiro_wilk" in tests:
        print(stats.get("s-w"))

    if "kurtosis" in tests:
        print(stats.get("kurt"))

    if "mean" in tests:
        print(f"Mean: {np.mean(probabilities):.4f}")

    if "standard_deviation" in tests:
        print(f"Standard deviation: {np.std(probabilities):.4f}")

    if "bimodal" in tests:
        bimodality_coeff, dip = test_bimodality(probabilities)
        print(f"Bimodality coefficient: {bimodality_coeff:.4f}")
        print(f"Dip: {dip}")

    if not load_probabilities:
        save_token_probabilities(probabilities, save_path=f'binary_data/Engine: {engine} Iterations: {iterations} Demo Sequence Length: {demo_sequence_length} Length: {length}.pt')

    if plot:
        plot_token_probabilities_histogram(probabilities, show=False, save_path=f'plots/Engine: {engine} Iterations: {iterations} Demo Sequence Length: {demo_sequence_length} Length: {length}.png')

if __name__ == "__main__":
    engine_options = ["ada", "babbage", "curie", "davinci", "text-ada-001", "text-babbage-001", "text-curie-001", "text-davinci-002", "text-davinci-003"]
    parser = argparse.ArgumentParser(description="Analyze probabilities from different OpenAI engines.")
    
    parser.add_argument("-e", "--engine", nargs="+", choices=engine_options + ["all"], required=True, help="Choose an OpenAI engine.")
    parser.add_argument("-i", "--iterations", type=int, required=True, help="Number of iterations.")
    parser.add_argument("-d", "--demo_sequence_length", type=int, required=True, help="Length of the demo sequence.")
    parser.add_argument("-l", "--length", type=int, required=True, help="Length of the generated sequences.")
    parser.add_argument("--generate", action="store_true", help="Generate probabilities instead of loading them.")
    parser.add_argument("--mean", action="store_true", help="Run mean test.")
    parser.add_argument("--sd", action="store_true", help="Run standard deviation test.")
    parser.add_argument("--ci", action="store_true", help="Run confidence interval test.")
    parser.add_argument("--normal", action="store_true", help="Run normality test.")
    parser.add_argument("--kurtosis", action="store_true", help="Run kurtosis test.")
    parser.add_argument("--valid", action="store_true", help="Run valid proportion test.")
    parser.add_argument("--shapiro", action="store_true", help="Run Shapiro-Wilk test.")
    parser.add_argument("--bimodal", action="store_true", help="Run bimodality test.")
    parser.add_argument("--plot", action="store_true", help="Plot histogram of probabilities.")
    
    args = parser.parse_args()
    
    engines = engine_options if "all" in args.engine else args.engine

    if len(engines) == 1:
        main(engine=engines[0], iterations=args.iterations, demo_sequence_length=args.demo_sequence_length, length=args.length, load_probabilities=not args.generate, plot=args.plot, tests=[test for test in ["mean", "sd", "ci", "normal", "kurtosis", "valid", "shapiro"] if vars(args).get(test)])
    elif len(engines) > 1:
        for engine in engines:
            print(f"Engine: {engine}")
            main(engine=engine, iterations=args.iterations, demo_sequence_length=args.demo_sequence_length, length=args.length, load_probabilities=not args.generate, plot=args.plot, tests=[test for test in ["mean", "sd", "ci", "normal", "kurtosis", "valid", "shapiro", "bimodal"] if vars(args).get(test)])
    else:
        print("No engine selected.")
