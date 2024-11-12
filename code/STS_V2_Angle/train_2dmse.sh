#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 angle_emb/angle_trainer.py \
--train_name_or_path SeanLee97/nli_for_simcse \
--save_dir v2/2d \
--model_name_or_path bert-base-uncased \
--tokenizer_name bert-base-uncased \
--pooling_strategy cls \
--maxlen 128 \
--ibn_w 30.0 \
--cosine_w 0.0 \
--apply_ese 1 \
--ese_kl_temperature 1.0 \
--ese_compression_size 128 \
--angle_w 1.0 \
--angle_tau 20.0 \
--learning_rate 5e-5 \
--logging_steps 100 \
--save_steps 200 \
--warmup_steps 50 \
--workers 128 \
--batch_size 128 \
--seed 42 \
--gradient_accumulation_steps 128 \
--epochs 10 \
--fp16 1

