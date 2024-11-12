#!/bin/bash


model_name=$1
layer_num_cut=$2
dim=$3

# model name replace("/" to "-")
model_name_save=$(echo $model_name | sed 's/\//-/g')

CUDA_VISIBLE_DEVICES=0 python3 angle_emb/angle_trainer.py \
--train_name_or_path SeanLee97/nli_for_simcse \
--save_dir v2/baselines/layer_${layer_num_cut}_dim_${dim} \
--model_name_or_path $model_name \
--tokenizer_name bert-base-uncased \
--pooling_strategy cls \
--layer_num_cut $layer_num_cut \
--embedding_size $dim \
--maxlen 128 \
--ibn_w 30.0 \
--cosine_w 0.0 \
--angle_w 1.0 \
--angle_tau 20.0 \
--learning_rate 5e-5 \
--logging_steps 100 \
--save_steps 200 \
--workers 16 \
--warmup_steps 50 \
--batch_size 128 \
--seed 42 \
--gradient_accumulation_steps 128 \
--epochs 10 \
--fp16 1
