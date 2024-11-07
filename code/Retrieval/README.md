# 2DMSE-Reproduce for Retrieval
Reproduction code of 2DMSE Retrieval results


## Code for Reproduction

### To train separate BERT models with different siezes:
For example to train 6 layer query encoder and 6 layer passage encoder with 128 dim:
```bash
Q=6
D=6
DIM=128
# if dim is not provided, set it to 768
RETRIEVER_OUTPUT_DIR=checkpoints/bert/retriever/retriever-bert-base-uncased_"$Q"_"$D"_dim"$DIM"-ep3-1e-4
mkdir -p $RETRIEVER_OUTPUT_DIR

python3 train_seperate.py \
  --output_dir $RETRIEVER_OUTPUT_DIR \
  --model_name_or_path bert-base-uncased \
  --seperated_encoder \
  --query_encoder_num_layers $Q \
  --passage_encoder_num_layers $D \
  --dim $DIM \
  --save_steps 10000 \
  --dataset_name Tevatron/msmarco-passage \
  --bf16 \
  --per_device_train_batch_size 128 \
  --train_group_size 8 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 196 \
  --num_train_epochs 3 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --report_to wandb \
  --run_name bert-base-uncased_"$Q"_"$D"_dim"$DIM"_ep3-1e-4
```

### To train 2DMSE model:
To train 2DMSE baseline model (original 2DMSE model):

```bash
score_align=False
compare_to_last_layer=False
sub_layer_full_dim=False
passage_to_last_layer=False
embedding_dim_list=128

output_dir=checkpoints/bert/retriever/retriever-bert-base-uncased_2dmse_dim
run_name=retriever-bert-base-uncased_2dmse_dim
output_dir=${output_dir}_${embedding_dim_list}
run_name=${run_name}_${embedding_dim_list}

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
fi

if [ $passage_to_last_layer = "True" ]; then
  output_dir=${output_dir}_passage_to_last_layer
  run_name=${run_name}_passage_to_last_layer
fi

python3 train_bert_russian_doll_v2.py \
  --output_dir ${output_dir} \
  --model_name_or_path bert-base-uncased \
  --tokenizer_name bert-base-uncased \
  --russian_doll_training \
  --save_steps 500 \
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
  --embedding_dim_list ${embedding_dim_list} \
  --run_name ${run_name}

```
Where
- `score_align` is whether to use logit score distribution for KL (False means use embedding PCA).
- `compare_to_last_layer` is whether compute KL to the last layer.
- `sub_layer_full_dim` is whether to add the full dimension of the sub-layer into the loss function.
- `passage_to_last_layer` to fix the document layer to the last layer (full layer).
- `embedding_dim_list` is the list of embedding dimensions for computing the loss.

### Evaluation

#### Encoding the dataset
Use the model checkpoint obtained above to encode the dataset. In this example we use MS MARCO dataset.
```bash
CHECKPOINT=YOU_CHECKPOINT

OUTPUT_DIR=embeddings/${CHECKPOINT}
mkdir -p ${OUTPUT_DIR}/msmarco

# encode query
query=dev # or dl19, dl20
python3 encode_russain_doll.py \
  --output_dir=temp \
  --model_name_or_path ${CHECKPOINT} \
  --tokenizer_name bert-base-uncased \
  --bf16 \
  --pooling cls \
  --per_device_eval_batch_size 64 \
  --query_max_len 32 \
  --passage_max_len 196 \
  --dataset_name Tevatron/msmarco-passage \
  --dataset_split ${query} \
  --encode_output_path ${OUTPUT_DIR}/msmarco/query.${query}.pkl \
  --encode_is_query \
  --layers_to_save 2,4,6,8,10,12
  
# encode corpus
python3 encode_russain_doll.py \
  --output_dir=temp \
  --model_name_or_path ${CHECKPOINT} \
  --tokenizer_name bert-base-uncased \
  --bf16 \
  --pooling cls \
  --per_device_eval_batch_size 64 \
  --query_max_len 32 \
  --passage_max_len 196 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --dataset_number_of_shards 1 \
  --dataset_shard_index 0 \
  --encode_output_path ${OUTPUT_DIR}/msmarco/corpus.${SHARD_INDEX}.pkl \
  --layers_to_save 2,4,6,8,10,12 \
  --layer_list 2,4,6,8,10,12 \
  --embedding_dim_list 32,64,128,256,512,768
```

We save the full size embedding of all the give layers.

#### Perform retrieval and evaluation
> Note: you need to install [pyserini](https://github.com/castorini/pyserini) for evaluation.
> 
For example, to perform retrieval and evaluation for the 6th layer with 128 dim:

```bash
n=6
dim=128

python search.py \
--query_reps ${OUTPUT_DIR}/msmarco/layer_"$n"/query.dev.pkl \
--passage_reps ${OUTPUT_DIR}/msmarco/layer_"$n"/"corpus*.pkl" \
--depth 1000 \
--batch_size 64 \
--save_text \
--save_ranking_to runs/run.dev.q"$n".d"$n".dim"$dim".txt \
--embedding_dim $dim

python -m tevatron.utils.format.convert_result_to_trec \
--input runs/run.dev.q"$n".d"$n".dim"$dim".txt \
--output runs/run.dev.q"$n".d"$n".dim"$dim".trec

python3 -m pyserini.eval.trec_eval -c -M 10 -m recip_rank msmarco-passage-dev-subset \
runs/run.dev.q"$n".d"$n".dim"$dim".trec
```