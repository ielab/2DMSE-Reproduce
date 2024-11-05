#!/bin/bash

OUTPUT_DIR=$1
RUN_DIR=$2
n=$3
dim=$4


CUDA_VISIBLE_DEVICES=0 python search.py \
--query_reps ${OUTPUT_DIR}/msmarco/layer_"$n"/query.dev.pkl \
--passage_reps ${OUTPUT_DIR}/msmarco/layer_"$n"/"corpus*.pkl" \
--depth 1000 \
--batch_size 64 \
--save_text \
--save_ranking_to ${RUN_DIR}/run.dev.q"$n".d"$n".dim"$dim".txt \
--embedding_dim $dim

python -m tevatron.utils.format.convert_result_to_trec \
--input ${RUN_DIR}/run.dev.q"$n".d"$n".dim"$dim".txt \
--output ${RUN_DIR}/run.dev.q"$n".d"$n".dim"$dim".trec

python3 -m pyserini.eval.trec_eval -c -M 10 -m recip_rank msmarco-passage-dev-subset \
${RUN_DIR}/run.dev.q"$n".d"$n".dim"$dim".trec > ${RUN_DIR}/run.dev.q"$n".d"$n".dim"$dim".eval

python3 -m pyserini.eval.trec_eval -c -m recall.100,1000 msmarco-passage-dev-subset \
${RUN_DIR}/run.dev.q"$n".d"$n".dim"$dim".trec >> ${RUN_DIR}/run.dev.q"$n".d"$n".dim"$dim".eval

