# Watermark Playground for CS 229BR S23

import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, LogitsProcessor

# torch.cuda.empty_cache()
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to("cuda")


def generate(
        prompt: str,
        length: int,
        num_beams: int = 1,
        repetition_penalty: float = 0.0001,
        logits_processors = [],
    ):  
    input_text = "This is a list of 50 ways a teenager could make money:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        input_ids,
        min_length=length,
        max_length=length,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        logits_processor=logits_processors,
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


class SingleLookbackWatermark(LogitsProcessor):
    """
    Beam Search watermark using a single token lookback.
    """

    def __init__(self, gamma: float, delta: float, soft=True):
        # Green list size gamma \in (0, 1)
        self.gamma = gamma
        assert 0 < gamma < 1

        # Hardness parameter \in [0, \infty]
        self.delta = delta

        # Hard or soft watermark 
        self.soft = soft


    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        # One previous token for each beam search beam
        prev_tokens = input_ids[:,-1]
        vocab_size = logits.shape[1]
        green_list_length = int(vocab_size * self.gamma)
        for i, token in enumerate(prev_tokens):
            if self.soft:
                random.seed(token)
                indices_to_mask = random.sample(range(vocab_size), green_list_length)
                logits[i, indices_to_mask] += self.delta
            else:
                # TODO(ltang): implement hard watermark
                pass

        return logits


# TODO(ltang): wrap this in main

# print(
#     generate(
#         "You are a random number generating machine. You will generate 1000 random single digit numbers, and you will output them in this format: 0,5,2,4,6,7,2,1,7,8,9,2 and so on. Here are the numbers: 1,2,7,3,9,0,5,", 
#         300, 
#         logits_processors=[SingleLookbackWatermark(gamma=0.5, delta=10)]
#     )
# )

