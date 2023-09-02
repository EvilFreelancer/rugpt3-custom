import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

name = 'ai-forever/ruGPT-3.5-13B'
# name = 'ai-forever/mGPT-13B'

# Loading the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    name,
    device_map='auto',
    load_in_8bit=True,
    max_memory={0: f'{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB'},
)
tokenizer = AutoTokenizer.from_pretrained(name)

# Sample texts
system_prompt = (
    "Ты - модель искусственного интеллекта ruGPT-3.5 13B, которая очень хорошо следует инструкциям. "
    "Твоя задача - помогать пользователю и отвечать на вопросы. "
    "Будь внимательна и не делай ничего противозаконного."
)
bot_message = 'Привет, чем я могу помочь?'
message = "Напиши стихотворение о программистах"
prompt = f"### Система:\n{system_prompt}\n\n### Бот:\n{bot_message}\n\n### Пользователь:\n{message}\n\n### Бот:\n"

# Run text-generation pipeline
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Generate output
output = pipe(
    prompt,
    max_new_tokens=256,
    top_k=40,
    top_p=0.85,
    repetition_penalty=1.1,
    do_sample=True,
    use_cache=False,
)
print(output[0]['generated_text'])
