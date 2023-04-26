"""
Analyze the distribution of a single 'randomly generated' digit from a LLM
"""

# TODO(ltang): implement hard watermark, then soft watermark, on ONLY one token generation
# See how it shifts distribution

import json
import matplotlib.pyplot as plt
import numpy as np
import openai
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
    if decode == 'beam':
        print("decode is beam")
        # TODO(ltang): refactor multinomial/greedy into here
        outputs = model.generate(
            input_ids,
            min_length=length,
            max_new_tokens=length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processors,
        )
        # TODO(ltang): postprocess and check for first occurence of a number
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("GEnerated text", generated_text)
        return generated_text
    
    else: 
        print("decode is NOT beam")
        with torch.no_grad():
            for _ in range(length):
                print("decoding one time step")    
                logits = model(input_ids)
                if watermark:
                    mask = watermark(input_ids) 
                else:
                    mask = torch.zeros_like(logits[0][0][-1])
                
                # Mask bumps up certain tokens' probabilities in the vocabulary
                logits[0][0][-1] += mask
                probs = torch.softmax(logits[0][0][-1], dim=0)
                
                if decode == 'multinomial':
                    sample = torch.multinomial(probs, 1)
                elif decode == 'greedy':
                    sample = torch.argmax(probs)
                else:
                    raise Exception

                # TODO(ltang): think about if we need to consider shifting distribution of whitespace (e.g. \n, \t)
                print("Sample:", sample)
                if sample.item().isnumeric():
                    return int(sample.item())

                # Input window extends to include generated token
                # Affects masker/watermark via `hash_tensor``
                input_ids = torch.cat([input_ids, sample.unsqueeze(0)], dim=1)

            # No digit produced. Just return number for now.
            return -50


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


def plot_digit_frequency(digits):

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

    with open('digit_counts.json', 'w') as file:
        json.dump(digit_counts, file, indent=4)
        
    plt.hist(digits, bins=100, range=(0, 100), alpha=0.7, density=True)
    plt.xlabel('Digit')
    plt.ylabel('Frequency')
    plt.title(f'Digit Frequencies for {len(digits)} Samples')
    plt.legend()
    plt.savefig('digit_freq.png')


def repeatedly_sample(prompt, model_name, engine="text-davinci-003", decode='beam', length=10, repetitions=2000):

    if model_name == 'openai-api':
        tokenizer = None
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
        d = generate_random_digit(prompt, tokenizer, model_name, model=model, length=length, decode=decode, engine=engine)
        sampled_digits.append(d)
    
    return sampled_digits


if __name__ == "__main__":
    # prompt = "What is a random value between 1 and 100?"
    # Alpaca Prompt
    prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    Generate a random number between 1 and 100.

    ### Response:"""
    digit_sample = repeatedly_sample(prompt, 'alpaca-lora', engine=None, decode='beam', length=500, repetitions=100)
    plot_digit_frequency(digit_sample)

