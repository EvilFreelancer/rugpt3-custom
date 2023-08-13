#!/bin/bash

[ ! -f pretrain_gpt3.py ] && wget https://raw.githubusercontent.com/EvilFreelancer/ru-gpts/master/pretrain_gpt3.py

output_dir="./rugpt3xl_custom"

python3 pretrain_gpt3.py \
  --train-data-path "./data/train.list" \
  --test-data-path "./data/valid.list" \
  --load-huggingface "ai-forever/rugpt3xl" \
  --logging-dir="${output_dir}/log/" \
  --save "${output_dir}/model" \
  --save-interval 1000 \
  --num-layers 16 \
  --hidden-size 2048 \
  --num-attention-heads 16 \
  --batch-size 1 \
  --seq-length 2048 \
  --max-position-embeddings 2048 \
  --train-iters 10000 \
  --distributed-backend nccl \
  --lr 0.0002 \
  --lr-decay-style cosine \
  --weight-decay 1e-2 \
  --warmup .01 \
  --log-interval 50 \
  --fp16 \
  --checkpoint-activations \
  --sparse-mode alternating
