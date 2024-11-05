#!/bin/bash

score_align=$1
compare_to_last_layer=$2
passage_to_last_layer=$3
sub_layer_full_dim=$4


output_dir=checkpoints/bert/retriever/retriever-bert-base-uncased_2dmse_dim128
run_name=retriever-bert-base-uncased_2dmse_dim128

if [ $score_align = "True" ]; then
  output_dir=${output_dir}_score
  run_name=${run_name}_score
fi

if [ $compare_to_last_layer = "True" ]; then
  output_dir=${output_dir}_last
  run_name=${run_name}_last
fi

if [ $sub_layer_full_dim = "True" ]; then
  output_dir=${output_dir}_sub_layer_full_dim
  run_name=${run_name}_sub_layer_full_dim

if [ $passage_to_last_layer = "True" ]; then
  output_dir=${output_dir}_passage_to_last_layer
  run_name=${run_name}_passage_to_last_layer
fi


CUDA_VISIBLE_DEVICES=0 python3 train_bert_russian_doll_v2.py \
  --output_dir ${output_dir} \
  --model_name_or_path bert-base-uncased \
  --tokenizer_name bert-base-uncased \
  --russian_doll_training \
  --save_steps 10000 \
  --dataset_name Tevatron/msmarco-passage \
  --pooling cls \
  --gradient_checkpointing \
  --per_device_train_batch_size 128 \
  --train_group_size 8 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 196 \
  --num_train_epochs 3 \
  --target_embedding_dim 128 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --report_to wandb \
  --score_align ${score_align} \
  --passage_to_last_layer ${passage_to_last_layer} \
  --compare_to_last_layer ${compare_to_last_layer} \
  --sub_layer_full_dim ${sub_layer_full_dim} \
  --run_name ${run_name}



