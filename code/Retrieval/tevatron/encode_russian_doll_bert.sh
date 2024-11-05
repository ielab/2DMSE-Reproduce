#!/bin/bash

CHECKPOINT=$1

OUTPUT_DIR=embeddings/${CHECKPOINT}
mkdir -p ${OUTPUT_DIR}/msmarco

#CUDA_VISIBLE_DEVICES=0 python encode_russain_doll.py \
#  --output_dir=temp \
#  --model_name_or_path ${CHECKPOINT} \
#  --tokenizer_name bert-base-uncased \
#  --bf16 \
#  --pooling cls \
#  --per_device_eval_batch_size 64 \
#  --query_max_len 32 \
#  --passage_max_len 196 \
#  --dataset_name Tevatron/msmarco-passage \
#  --dataset_number_of_shards 1 \
#  --dataset_shard_index 0 \
#  --encode_output_path ${OUTPUT_DIR}/msmarco/corpus.0.pkl \
#  --layers_to_save 12 \
#  --layer_list 12 \
#  --embedding_dim_list 32,64,128,256,512,768


CUDA_VISIBLE_DEVICES=0 python encode_russain_doll.py \
  --output_dir=temp \
  --model_name_or_path ${CHECKPOINT} \
  --tokenizer_name bert-base-uncased \
  --bf16 \
  --pooling cls \
  --per_device_eval_batch_size 64 \
  --query_max_len 32 \
  --passage_max_len 196 \
  --dataset_name Tevatron/msmarco-passage \
  --dataset_split dev \
  --encode_is_query \
  --encode_output_path ${OUTPUT_DIR}/msmarco/query.pkl \
  --layers_to_save 12 \
  --layer_list 12 \
  --embedding_dim_list 32,64,128,256,512,768