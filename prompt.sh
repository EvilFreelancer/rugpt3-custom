#!/bin/bash

[ ! -f generate_transformers.py ] && wget https://raw.githubusercontent.com/ai-forever/ru-gpts/master/generate_transformers.py

python3 generate_transformers.py \
  --model_type=gpt2 \
  --model_name_or_path=dostoevsky_doesnt_write_it \
  --k=5 \
  --p=0.95 \
  --length=100
