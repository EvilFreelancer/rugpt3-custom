#!/bin/bash

[ ! -f generate_samples.py ] && wget https://raw.githubusercontent.com/EvilFreelancer/ru-gpts/master/generate_samples.py

python3 generate_samples.py \
  --num-workers 1 \
  --fp16 \
  --num-layers 16 \
  --hidden-size 2048 \
  --num-attention-heads 16 \
  --seq-length 1024 \
  --out-seq-length 1024 \
  --max-position-embeddings 2048 \
  --sparse-mode alternating \
  --tokenizer-path "ai-forever/rugpt3xl" \
  --load "./rugpt3xl_custom/model" \
  --top_k 5 \
  --top_p 0.95
