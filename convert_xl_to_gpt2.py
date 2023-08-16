import os
import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, pipeline

if not os.path.exists("ru_gpts"):
    raise Exception(
        "Folder `ru_gpts` not found! Please run `git clone https://github.com/EvilFreelancer/ru-gpts.git ru_gpts` first")

from ru_gpts.src.xl_wrapper import RuGPT3XL


def convert_gpt3_to_gpt2(rugpt3xl_model, gpt2_model):
    """Convert RuGPT3XL's weights to GPT2LMHeadModel."""

    # Word embeddings
    gpt2_model.transformer.wte.weight.data = rugpt3xl_model.model.module.word_embeddings.weight.data

    # Position embeddings
    gpt2_model.transformer.wpe.weight.data = rugpt3xl_model.model.module.position_embeddings.weight.data

    # Transformer layers
    for gpt3_block, gpt2_block in zip(rugpt3xl_model.model.module.transformer.layers, gpt2_model.transformer.h):
        # Attention weights
        gpt2_block.attn.c_attn.weight.data = gpt3_block.attention.query_key_value.weight.data.t()
        gpt2_block.attn.c_attn.bias.data = gpt3_block.attention.query_key_value.bias.data
        gpt2_block.attn.c_proj.weight.data = gpt3_block.attention.dense.weight.data.t()
        gpt2_block.attn.c_proj.bias.data = gpt3_block.attention.dense.bias.data

        # MLP weights
        gpt2_block.mlp.c_fc.weight.data = gpt3_block.mlp.dense_h_to_4h.weight.data.t()
        gpt2_block.mlp.c_fc.bias.data = gpt3_block.mlp.dense_h_to_4h.bias.data
        gpt2_block.mlp.c_proj.weight.data = gpt3_block.mlp.dense_4h_to_h.weight.data.t()
        gpt2_block.mlp.c_proj.bias.data = gpt3_block.mlp.dense_4h_to_h.bias.data

        # LayerNorm weights
        gpt2_block.ln_1.weight.data = gpt3_block.input_layernorm.weight.data.t()
        gpt2_block.ln_1.bias.data = gpt3_block.input_layernorm.bias.data
        gpt2_block.ln_2.weight.data = gpt3_block.post_attention_layernorm.weight.data.t()
        gpt2_block.ln_2.bias.data = gpt3_block.post_attention_layernorm.bias.data

    return gpt2_model


def test_layers(rugpt3xl_model, gpt2_model, input_tensor):
    """Tests the activations of each layer for two models given the same input tensor."""

    with torch.no_grad():
        # Initial embeddings
        gpt3xl_embedding = rugpt3xl_model.model.module.word_embeddings(input_tensor)
        gpt2_embedding = gpt2_model.transformer.wte(input_tensor)

        if not torch.allclose(gpt3xl_embedding, gpt2_embedding, atol=1e-6):
            return "Embedding layers do not match."

        gpt3xl_hidden = gpt3xl_embedding
        gpt2_hidden = gpt2_embedding

        # Create ltor_mask for GPT3ParallelTransformerLayer
        ltor_mask = torch.triu(torch.ones(input_tensor.size(1), input_tensor.size(1)), diagonal=1).to('cuda:0').half()

        # Iterate over layers
        for layer_idx, (gpt3_block, gpt2_block) in enumerate(
                zip(rugpt3xl_model.model.module.transformer.layers, gpt2_model.transformer.h)):
            # Pass through one transformer block
            gpt3xl_hidden = gpt3_block(gpt3xl_hidden, ltor_mask)[0]
            gpt2_hidden = gpt2_block(gpt2_hidden)[0]

            if not torch.allclose(gpt3xl_hidden, gpt2_hidden, atol=1e-6):
                print(f"Difference found in layer {layer_idx}.")

            # Вывод выходных данных каждого слоя для проверки (можно убрать этот блок, если не требуется вывод)
            print(f"Layer {layer_idx} outputs (RuGPT3XL and GPT2):")
            print(gpt3xl_hidden)
            print(gpt2_hidden)
            print("----------------------")

    return "All layers match."


name = 'ai-forever/rugpt3xl'

# Initialize GPT3Model (assuming you have already done so)
rugpt3xl_model = RuGPT3XL.from_pretrained(name)
rugpt3xl_tokenizer = GPT2Tokenizer.from_pretrained(name)

# Load pre-initialized GPT2LMHeadModel
gpt2_config = GPT2Config(
    vocab_size=rugpt3xl_model.model.module._conf_dict["vocab_size"],
    n_positions=rugpt3xl_model.model.module._conf_dict["n_positions"],
    n_ctx=rugpt3xl_model.model.module._conf_dict["n_ctx"],
    n_embd=rugpt3xl_model.model.module._conf_dict["n_embd"],
    n_layer=rugpt3xl_model.model.module._conf_dict["n_layer"],
    n_head=rugpt3xl_model.model.module._conf_dict["n_head"],
)
gpt2_model = GPT2LMHeadModel(gpt2_config)

# pipe = pipeline(
#     'text-generation',
#     model=gpt2_model,
#     tokenizer=rugpt3xl_tokenizer,
#     device='cuda:0',
# )
# prompt_text = "Москва"
# output = pipe(
#     prompt_text,
#     max_new_tokens=1024,
#     top_k=20,
#     top_p=0.9,
#     repetition_penalty=1.1,
#     temperature=1.0,
#     # seed=42,
#     do_sample=True,
#     use_cache=False
# )
# print(output)
# exit()

# Convert and transfer weights
gpt2_model = convert_gpt3_to_gpt2(rugpt3xl_model, gpt2_model)

# Test layers
input_tensor = torch.randint(0, rugpt3xl_tokenizer.vocab_size, (1, 10)).to('cuda:0')
print(test_layers(rugpt3xl_model.to('cuda:0'), gpt2_model.to('cuda:0'), input_tensor))

# Save the GPT2 model
output = './converted_gpt2_model'
gpt2_model.save_pretrained(output)
rugpt3xl_tokenizer.save_pretrained(output)
