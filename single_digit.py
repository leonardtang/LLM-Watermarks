"""
Analyze the distribution of a single 'randomly generated' digit from a LLM
"""

# TODO(ltang): implement hard watermark, then soft watermark, on ONLY one token generation
# See how it shifts distribution

import json
import matplotlib.pyplot as plt
import numpy as np
import openai
import random
import torch
from api_keys import OPENAI_API_KEY
from collections import defaultdict
from watermark_playground import SingleLookbackWatermark
from transformers import GPT2Tokenizer, GPT2LMHeadModel, LlamaTokenizer, LlamaForCausalLM

openai.api_key = OPENAI_API_KEY
device = torch.device("cuda:4") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


def generate_from_model(
    model, 
    input_ids, 
    tokenizer,
    length: int,
    decode: str = 'beam',
    num_beams=4,
    repetition_penalty: float = 0.0001,
    logits_processors = [],
):
    # Beam search multinomial sampling
    if decode == 'beam':
        print("decode is beam")
        beam_count = num_beams
        do_sample = True
    elif decode == 'multinomial':
        print("decode is multi")
        beam_count = 1
        do_sample = True
    elif decode == 'greedy':
        print("decode is greedy")
        beam_count = 1
        do_sample = False
    else:
        raise Exception

    outputs = model.generate(
        input_ids,
        min_length=length,
        max_new_tokens=length,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=logits_processors,
        do_sample=True,
    )
    # TODO(ltang): postprocess and check for first occurence of a number
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated text", generated_text)
    return generated_text


def generate_random_digit(
    prompt, 
    tokenizer, 
    model_name='openai-api', 
    model=None, 
    engine='text-davinci-003', 
    length=10,
    # Watermarking Callable
    watermark=None,
    decode='beam',
):
    """
    - 'Randomly' generate a digit from a given range (not necessarily binary)
    - Assume no access to logits for now
    - See prompt sources in `prompt_list.txt`
    """

    assert decode in ['beam', 'greedy', 'multinomial']

    if model_name == 'openai-api':
        # TODO(ltang): imeplement Watemarks using logit_bias on OpenAI model
        # Recover the gamma params and whatnot
        if watermark:
            logit_bias = {}
            # TODO(ltang): figure out what the vocab size is actually supposed to be for each OpenAI model
            # https://enjoymachinelearning.com/blog/the-gpt-3-vocabulary-size/
            vocab_size = 175000
            green_list_length = int(vocab_size * watermark.gamma)
            indices_to_mask = random.sample(range(vocab_size), green_list_length)
            # Note that logit_bias persists for the entire generation.
            # So potentially want to re-feed in prompt
            for idx in indices_to_mask:
                logit_bias[idx] = watermark.delta
        else:
            logit_bias = None
        while True:
            try:
                response = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    # Though we want to generate a single digit, there may be misc. characters like "\n" and such
                    max_tokens=length, 
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

    # TODO(ltang): Analyze digit distribution of an open-source model with and without watermark
    elif model_name == 'gpt-2':
        print("elif model_name == 'gpt-2':")
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
        processor = [watermark] if watermark else []
        generate_from_model(model, input_ids, tokenizer, length, decode, logits_processors=processor, repetition_penalty=1.3)
        
    elif model_name == 'alpaca-lora':
        print("elif model_name == 'alpaca-lora':")
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
        processor = [watermark] if watermark else []
        generate_from_model(model, input_ids, tokenizer, length, decode, logits_processors=processor, repetition_penalty=1)
    else: 
        pass


def plot_digit_frequency(digits, output_file):

    print("Raw digits!", digits)
    # Construct dict for numerical reference
    digit_counts = {}
    # TODO(ltang): don't hardcode this range in the future
    for i in range(100):
        digit_counts[i] = 0 

    for d in digits:
        if d in set(range(100)): 
            digit_counts[d] += 1
        else:
            digit_counts[d] = 500

    numbered_out_file = output_file.split('.')[0] + random.getrandbits(8) + '.json'
    with open(output_file, 'w') as file:
        json.dump(digit_counts, file, indent=4)
        
    plt.hist(digits, bins=100, range=(0, 100), alpha=0.7, density=True)
    plt.xlabel('Digit')
    plt.ylabel('Frequency')
    plt.title(f'Digit Frequencies for {len(digits)} Samples')
    plt.legend()
    png_file = numbered_out_file.split('.')[0] + '.png'
    plt.savefig(png_file)


def repeatedly_sample(prompt, model_name, engine="text-davinci-003", decode='beam', length=10, repetitions=2000, watermark=None):

    assert model_name in ['openai-api', 'gpt-2', 'alpaca-lora']
    
    if model_name == 'openai-api':
        tokenizer = None
        model = None
    elif model_name == 'gpt-2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2-xl').to(device)
        vocab_size = model.config.vocab_size
    elif model_name == 'alpaca-lora':
        tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
        model = LlamaForCausalLM.from_pretrained(
            "chainyo/alpaca-lora-7b",
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    
    print(f"Sampling for {repetitions} repetitions")
    sampled_digits = []
    for _ in range(repetitions):
        d = generate_random_digit(prompt, tokenizer, model_name, model=model, length=length, decode=decode, engine=engine, watermark=watermark)
        sampled_digits.append(d)
    
    return sampled_digits


if __name__ == "__main__":
    prompt = "Pick a random number between 1 and 100. Just return the number, don't include any other text or punctuation in the response."
    # prompt = "What is a random value between 1 and 100?"
    # Alpaca Prompt
    # alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    # ### Instruction:
    # Generate a random number between 1 and 100.

    # ### Response:"""

    watermark = SingleLookbackWatermark(gamma=0.5, delta=10)
    digit_sample = repeatedly_sample(prompt, 'openai-api', engine='text-davinci-003', decode='beam', length=10, repetitions=2000)
    plot_digit_frequency(digit_sample, 'digit_counts_td3_05_10.json')

