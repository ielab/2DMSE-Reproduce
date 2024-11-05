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
import sys
import traceback
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator, SimilarityFunction

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"



batch_size = 128
gradient_accumulation_steps = 128
num_train_epochs = 10
matryoshka_dims = [768, 512, 256, 128, 64, 32, 16, 8]

# Save path of the model
output_dir = f"{model_name.replace('/', '-')}/1d"

# 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
# create one with "mean" pooling.
model = SentenceTransformer(model_name)
# If we want, we can limit the maximum sequence length for the model
# model.max_seq_length = 75
logging.info(model)

# 2. Load the STSB dataset: https://huggingface.co/datasets/sentence-transformers/stsb
train_dataset = load_dataset("SeanLee97/nli_for_simcse", split="train")
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
logging.info(train_dataset)

# 3. Define our training loss
# CoSENTLoss (https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosentloss) needs two text columns and one
# similarity score column (between 0 and 1)
inner_train_loss = losses.MultipleNegativesRankingLoss(model)
train_loss = losses.MatryoshkaLoss(model, loss=inner_train_loss, matryoshka_dims=matryoshka_dims)



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
    # Optional tracking/debugging parameters:
    #eval_strategy="steps",
    #eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=1000,
    run_name="matryoshka-sts",  # Will be used in W&B if `wandb` is installed
)

# 6. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    #eval_dataset=eval_dataset,
    loss=train_loss,
    #evaluator=dev_evaluator,
)
trainer.train()


# 7. Evaluate the model performance on the STS Benchmark test dataset
evaluators = []
for dim in matryoshka_dims:
    evaluators.append(
        EmbeddingSimilarityEvaluator(
            sentences1=test_dataset["sentence1"],
            sentences2=test_dataset["sentence2"],
            scores=test_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name=f"sts-test-{dim}",
            truncate_dim=dim,
        )
    )
test_evaluator = SequentialEvaluator(evaluators)
test_evaluator(model)

# 8. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)

