"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with MatryoshkaLoss using MultipleNegativesRankingLoss. This trains a model at output dimensions [768, 512, 256, 128, 64].
Entailments are positive pairs and the contradiction on AllNLI dataset is added as a hard negative.
At every 10% training steps, the model is evaluated on the STS benchmark dataset

Usage:
python 2d_matryoshka_nli.py

OR
python 2d_matryoshka_nli.py pretrained_transformer_model_name
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
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers.training_args import BatchSamplers

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
batch_size = 128  # The larger you select this, the better the results (usually). But it requires more GPU memory
gradient_accumulation_steps = 128
num_train_epochs = 10

# Save path of the model
output_dir = f"{model_name.replace('/', '-')}/2d"

# 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
# create one with "mean" pooling.
model = SentenceTransformer(model_name)
# If we want, we can limit the maximum sequence length for the model
# model.max_seq_length = 75
logging.info(model)

# 2. Load the AllNLI dataset: https://huggingface.co/datasets/sentence-transformers/all-nli
train_dataset = load_dataset("SeanLee97/nli_for_simcse", split="train")
#eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
logging.info(train_dataset)

# If you wish, you can limit the number of training samples
# train_dataset = train_dataset.select(range(5000))

# 3. Define our training loss
inner_train_loss = losses.MultipleNegativesRankingLoss(model)
train_loss = losses.Matryoshka2dLoss(model, inner_train_loss, [768, 512, 256, 128, 64, 32, 16, 8])

# 5. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    warmup_steps=50,
    seed=42,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    #eval_strategy="steps",
    #eval_steps=1000,
    save_strategy="steps",
    #save_steps=1000,
    save_total_limit=2,
    #logging_steps=1000,
    run_name="2d-matryoshka-nli",  # Will be used in W&B if `wandb` is installed
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
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["sentence1"],
    sentences2=test_dataset["sentence2"],
    scores=test_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-test",
)
test_evaluator(model)

# 8. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)