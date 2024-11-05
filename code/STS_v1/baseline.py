"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch.
It uses MatryoshkaLoss with the powerful CoSENTLoss to train models that perform well at output dimensions [768, 512, 256, 128, 64].
It generates sentence embeddings that can be compared using cosine-similarity to measure the similarity.

Usage:
python matryoshka_sts.py

OR
python matryoshka_sts.py pretrained_transformer_model_name
"""

import logging
import os
import sys
import traceback
from datetime import datetime
from tqdm import tqdm

from datasets import load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator, SimilarityFunction


model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"



batch_size = 128
gradient_accumulation_steps = 128
num_train_epochs = 10

dims = []

layers = []

train_dataset = load_dataset("SeanLee97/nli_for_simcse", split="train")

output_parent= f"v1/baselines"


for i in [2, 4, 6, 8, 10, 12]:
    for j in [8, 16, 32, 64, 128, 256, 512, 768]:
        if os.path.exists(f"{output_parent}/layer_{i}_dim_{j}/final"):
            continue
        dims.append(j)
        layers.append(i)



test_dataset = load_dataset("sentence-transformers/stsb", split="test")


for layer_idx, layer in enumerate(layers):
    # for dim in dims:
    # Save path of the model
    dim = dims[layer_idx]
    output_dir = f"{output_parent}/sts_bert-base-uncased_layer_{layer}_dim_{dim}"

    # 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
    # create one with "mean" pooling.
    model = SentenceTransformer(model_name)
    model._first_module().auto_model.encoder.layer = model._first_module().auto_model.encoder.layer[:layer]
    # If we want, we can limit the maximum sequence length for the model
    # model.max_seq_length = 75
    logging.info(model)

    # 2. Load the STSB dataset: https://huggingface.co/datasets/sentence-transformers/stsb

    logging.info(train_dataset)

    # 3. Define our training loss
    # CoSENTLoss (https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosentloss) needs two text columns and one
    # similarity score column (between 0 and 1)

    train_loss = losses.MultipleNegativesRankingLoss(model)


    # 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.


    # 5. Define the training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        # Optional training parameters:
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=50,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        logging_steps=1000,
        run_name=f"sts-{layer}-{dim}",  # Will be used in W&B if `wandb` is installed
    )

    # 6. Create the trainer & start training
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,

    )
    trainer.train()

    # 8. Save the trained & evaluated model locally
    final_output_dir = f"{output_dir}/final"
    model.save(final_output_dir)
