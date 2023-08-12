#!/bin/bash

[ ! -f pretrain_transformers.py ] && wget https://raw.githubusercontent.com/ai-forever/ru-gpts/master/pretrain_transformers.py

python3 pretrain_transformers.py \
  --output_dir=dostoevsky_doesnt_write_it \
  --model_type=gpt2 \
  --model_name_or_path=sberbank-ai/rugpt3small_based_on_gpt2 \
  --do_train \
  --train_data_file=data/train.txt \
  --do_eval \
  --fp16 \
  --eval_data_file=data/valid.txt \
  --per_gpu_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 5 \
  --block_size 512 \
  --overwrite_output_dir
