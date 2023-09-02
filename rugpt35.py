import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = 'ai-forever/ruGPT-3.5-13B'

# Loading the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    name,
    device_map='auto',
    load_in_8bit=True,
    max_memory={0: f'{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB'}
)
tokenizer = AutoTokenizer.from_pretrained(name)

# Sample texts
system_prompt = (
    "### System:\nYou are an AI that follows instructions extremely well. Help as much as you can. "
    "Remember, be safe, and don't do anything illegal.\n\n"
)
message = "Write me a poem please"
prompt = f"{system_prompt}### User: {message}\n\n### Assistant: "

# Generate output
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
