import torch
import time
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

name = './converted_gpt2_model'

# Start model load time
start_time = time.time()

# Loading the model and tokenizer
model = GPT2LMHeadModel.from_pretrained(name).cuda()
tokenizer = GPT2Tokenizer.from_pretrained(name)

# Run text-generation pipeline
pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    device='cuda:0',
)

# Model load time
model_load_time = time.time() - start_time

# Sample texts
# system_prompt = "### Система:\nВы искусственный интеллект, который отлично следует инструкциям. Помогайте насколько сможете. Помните, будьте осторожны и не делайте ничего незаконного.\n\n"
# message = "Напишите мне стих, пожалуйста"
# prompt_text = f"{system_prompt}### Пользователь: {message}\n\n### Ассистент: "
prompt_text = f"Москва"
input_ids = tokenizer.encode(prompt_text, return_tensors="pt").cuda()

# Start generation time
generation_start_time = time.time()

with torch.no_grad():
    output = pipe(
        prompt_text,
        max_new_tokens=1024,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        temperature=1.0,
        # seed=42,
        do_sample=True,
        use_cache=False
    )

generated_text = output[0]['generated_text']
cleaned_text = re.sub(r'[\xa0\xad“«»\-\\\n]+', ' ', generated_text)
cleaned_text = ' '.join(cleaned_text.split())

print(cleaned_text)
