#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=100G
#SBATCH --job-name=TREC_RAG
#SBATCH --partition=gpu_cuda
#SBATCH --account=a_ielab
#SBATCH --gres=gpu:h100:1
#SBATCH --time=10:00:00
#SBATCH -o /scratch/project/neural_ir/dylan/bertLlama/print.txt
#SBATCH -e /scratch/project/neural_ir/dylan/bertLlama/error.txt

export WANDB_PROJECT=RRepllama
module load cuda
module load gcc

source activate /scratch/project/neural_ir/dylan/env/bertllama


input_dir="/scratch/project/neural_ir/dylan/bertLlama/Angie/v2"
output_dir="/scratch/project/neural_ir/dylan/bertLlama/Angie/v2_converted"

# first start with baseline
# layer is 2, 4, 6, 8, 10, 12
# dim is 32, 64, 128, 256, 512, 768

for layer_num in 2 4 6 8 10 12; do
    for dim in 8 16; do
        echo "Converting layer $layer_num with dim $dim"
        current_output_dir=$output_dir/baselines/layer_${layer_num}_dim_${dim}/final
        mkdir -p $current_output_dir

        python convert_to_sentence_transformer.py \
        --model_name_or_path $input_dir/baselines/layer_${layer_num}_dim_${dim} \
        --output_dir $current_output_dir \
        --pooling_strategy cls
    done
done

# now it's the 2d one
#python3 convert_to_sentence_transformer.py \
#--model_name_or_path $input_dir/2d \
#--output_dir $output_dir/2d \
#--pooling_strategy cls

# now it's the 1d one
#python3 convert_to_sentence_transformer.py \
#--model_name_or_path $input_dir/1d \
#--output_dir $output_dir/1d \
#--pooling_strategy cls