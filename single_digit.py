"""
Analyze the distribution of a single 'randomly generated' digit from a LLM
"""

# TODO(ltang): implement hard watermark, then soft watermark, on ONLY one token generation
# See how it shifts distribution

import json
import matplotlib.pyplot as plt
import numpy as np
import openai
from collections import defaultdict

openai.api_key = openai.key

def generate_random_digit(prompt, model='openai-api', engine='text-davinci-003'):
    """
    - 'Randomly' generate a digit from a given range (not necessarily binary)
    - Assume no access to logits for now
    - See prompt sources in `prompt_list.txt`
    """

    if model == 'openai-api':
        while True:
            try:
                response = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    # Only want to generate a single digit
                    max_tokens=10, 
                    temperature=1,
                    logprobs=2,
                )

                raw_digit = response.choices[0].text.strip()
                int_digit = int(raw_digit)
            except:
                # Catch when generated `raw_digit` is not of `int` type
                # Or just catch general OpenAI server error 
                continue
            else:
                break
        
        return int_digit

    # TODO(ltang): Implement open-source model with and without watermark
    elif model == 'gpt-2':
        pass
    else: 
        pass


def plot_digit_frequency(digits):

    # Construct dict for numerical reference
    digit_counts = {}
    # TODO(ltang): don't hardcode this range in the future
    for i in range(100):
        digit_counts[i] = 0 

    for d in digits: 
        digit_counts[d] += 1

    with open('digit_counts.json', 'w') as file:
        json.dump(digit_counts, file, indent=4)
        
    plt.hist(digits, bins=100, range=(0, 100), alpha=0.7, density=True)
    plt.xlabel('Digit')
    plt.ylabel('Frequency')
    plt.title(f'Digit Frequencies for {len(digits)} Samples')
    plt.legend()
    plt.savefig('digit_freq.png')


def repeatedly_sample(prompt, engine="text-davinci-003", repetitions=2000):
    
    print(f"Sampling for {repetitions} repetitions")
    sampled_digits = []
    for _ in range(repetitions):
        d = generate_random_digit(prompt)
        sampled_digits.append(d)
    
    return sampled_digits


if __name__ == "__main__":
    prompt = "Pick a random number between 1 and 100. Just return the number, don't include any other text or punctuation in the response."
    digit_sample = repeatedly_sample(prompt, repetitions=2000)
    plot_digit_frequency(digit_sample)


