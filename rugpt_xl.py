import os
import torch
from transformers import GPT2Tokenizer
import time

if not os.path.exists("ru_gpts"):
    raise Exception("Folder `ru_gpts` not found! Please run `git clone https://github.com/EvilFreelancer/ru-gpts.git ru_gpts` first")

from ru_gpts.src.xl_wrapper import RuGPT3XL

name = 'ai-forever/rugpt3xl'

start_time = time.time()  # Start model load time

# Download model source and weights
model = RuGPT3XL.from_pretrained(name, seq_len=1024)

# Download tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(name)

model_load_time = time.time() - start_time  # Model load time

system_prompt = "### System:\nYou are an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n"
message = "Write me a poem please"
prompt = f"{system_prompt}### User: {message}\n\n### Assistant: "

generation_start_time = time.time()  # Start generation time

with torch.autocast('cuda'):
    output = model.generate(
        prompt,
        max_length=1024,
        do_sample=True,
        top_k=20,
        top_p=0.95,
        repetition_penalty=1.1,
        early_stopping=False,
        num_beams=1,
        num_beam_groups=1,
        num_return_sequences=1,
        temperature=1.0,
    )
    print(output)

generation_time = time.time() - generation_start_time  # Generation time

tokens_per_second = len(tokenizer.encode(output[0])) / generation_time  # Tokens per second

# Print results
print(f"Model loading time: {model_load_time:.2f} seconds")
print(f"Generation time: {generation_time:.2f} seconds")
print(f"Tokens per second: {tokens_per_second:.1f}")

# time.sleep(10)
