import torch
import logging
from .encoder import EncoderModel, EncoderOutput
from .dense import DenseModel
from transformers import PreTrainedModel, AutoModel
from typing import Dict, Optional
import copy
from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import os
from typing import Dict, List
from torch import nn, Tensor
import numpy as np
import math
import random
import torch.distributed as dist
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class DenseAdaptiveRussianDollModel(DenseModel):

    def __init__(self,
                 kl_divergence_weight: float = 0.0,
                 layer_list: List = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.kl_divergence_weight = kl_divergence_weight
        if layer_list is not None:
            self.layer_list = [int(i) - 1 for i in layer_list]
        else:
            self.layer_list = [-1]

        # add a linear layer for binary classification
        self.linear = nn.Linear(self.encoder.config.hidden_size, 1)




    def gradient_checkpointing_enable(self, **kwargs):
        # check if the encoder has gradient_checkpointing_enable method
        if hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()
        else:
            self.encoder.model.gradient_checkpointing_enable()


    def encode_query_train(self, qry):
        query_output = self.encoder(**qry, return_dict=True, output_hidden_states=True)
        all_hidden_states = torch.stack(query_output.hidden_states[1:])  # Stack all hidden states into a new dimension, except the first one which is the embedding layer
        all_reps = self._pooling(all_hidden_states, qry['attention_mask'])
        # then for each reps in all_reps, pass through the linear layer, if classification score is 0, then this is the returned rep, otherwise, pass through the next layer
        return all_reps

    def encode_passage_train(self, psg):
        # Encode passage is the same as encode query
        return self.encode_query_train(psg)


    def encode_query(self, qry):
        query_output = self.encoder(**qry, return_dict=True)
        query_hidden_states = torch.stack(query_output.hidden_states[1:])
        all_reps = self._pooling(query_hidden_states, qry['attention_mask'])
        for i, reps in enumerate(all_reps):
            if i not in self.layer_list:
                continue
            classification = self.linear(reps)
            if classification == 0:
                return reps
        return all_reps[-1]

    def encode_passage(self, psg):
        # Encode passage is the same as encode query
        return self.encode_query(psg)

    def _pooling(self, all_hidden_states, attention_mask):
        # all_hidden_states shape: [num_layers, batch_size, seq_length, hidden_dim]
        if self.pooling in ['cls', 'first']:
            # Select the first token for all layers
            reps = all_hidden_states[:, :, 0, :]  # Shape: [num_layers, batch_size, hidden_dim]
        elif self.pooling in ['mean', 'avg', 'average']:
            # Expand attention mask to match hidden states dimensions
            expanded_mask = attention_mask[:, None, :, None]  # Shape: [batch_size, 1, seq_length, 1]
            masked_hiddens = all_hidden_states.masked_fill(~expanded_mask.bool(), 0.0)
            reps = masked_hiddens.sum(dim=2) / attention_mask.sum(dim=1, keepdim=True)[:, None, None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            reps = all_hidden_states[:, torch.arange(all_hidden_states.size(1)), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        # reps dimensions: [num_layers, batch_size, hidden_dim]
        return reps

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=1)

        return all_tensors

    def pca_reduce(self, scores, target_dim):
        batch_size, sequence_length, dim = scores.shape
        N_samples = batch_size * sequence_length

        # Reshape to (N_samples, dim)
        scores_reshaped = scores.reshape(N_samples, dim)

        # Center the data by subtracting the mean
        mean = scores_reshaped.mean(dim=0, keepdim=True)
        scores_centered = scores_reshaped - mean

        # Compute SVD of the centered data
        U, S, Vh = torch.linalg.svd(scores_centered, full_matrices=False)
        # Vh has shape (dim, dim)

        # Select the top 'target_dim' components
        Vh_top = Vh[:target_dim, :]  # Shape: (target_dim, dim)

        # Project data onto top components
        scores_reduced = torch.matmul(scores_centered, Vh_top.T)  # Shape: (N_samples, target_dim)

        # Reshape back to (batch_size, sequence_length, target_dim)
        scores_reduced = scores_reduced.reshape(batch_size, sequence_length, target_dim)

        return scores_reduced

    def compute_dl_divergence(self, scores, target):
        return F.kl_div(F.log_softmax(scores / self.temperature, dim=1), F.softmax(target / self.temperature, dim=1),
                        reduction='batchmean')


    def compute_matryoshka_loss(self, full_dim_scores, target, target_scores):
        total_loss = 0
        scores = full_dim_scores.sum(dim=-1)
        total_loss += self.compute_loss(scores / self.temperature, target)
        if self.kl_divergence_weight > 0:
            kl_loss = self.compute_dl_divergence(scores, target_scores)
            total_loss += self.kl_divergence_weight * kl_loss
        return total_loss

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query_train(query) if query else None
        p_reps = self.encode_passage_train(passage) if passage else None
        #print("dim of q_reps: ", q_reps.shape)
        #print("dim of p_reps: ", p_reps.shape)
        # Early return for inference when either query or passage representations are missing
        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        # Initialize storage for scores and potential loss across all layers
        layer_max_scores = []
        total_loss = 0 if self.training else None


        if self.training:
            # hard code for now
            q_reps = q_reps[self.layer_list]
            p_reps = p_reps[self.layer_list]

            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            # Ensure that we are handling the dimensions correctly
            # Assume q_reps and p_reps are of shape [num_layers, batch_size, feature_dim]
            num_layers, batch_size, embedding_dim = q_reps.shape

            # first is to compute the similarity scores loss
            target = torch.arange(q_reps.size(1), device=q_reps.device, dtype=torch.long)
            target = target * (p_reps.size(1) // q_reps.size(1))
            all_dim_scores = torch.einsum('lqh,Ldh -> lLqdh', q_reps, p_reps)  # [num_layers, num_layers, batch_size, hidden_dim]

            # last layer score for query_reps and passage_rep is target for dl_divergence
            full_layer_full_dim_scores = all_dim_scores[-1, -1, :, :]
            target_score = full_layer_full_dim_scores.sum(dim=-1).detach()

            # get last layer rank for each target

            all_scores = all_dim_scores.sum(dim=-1)
            sorted_indices_desc = all_scores.argsort(dim=-1, descending=True) # now sorted_indices_desc is [num_layers, num_layers, batch_size_query, batch_size_passage]
            # now get the indices of target, for each indices in dimention 2

            final_tensor = torch.zeros((all_scores.size(0), all_scores.size(1), all_scores.size(2)),
                                       device=all_scores.device, dtype=torch.long) # shape [num_layers, num_layers, batch_size_query]
            # Iterate through each layer
            for i in range(all_scores.size(0)):
                for j in range(all_scores.size(1)):
                    # Select the appropriate indices from sorted_indices_desc using target
                    final_tensor[i, j, :] = sorted_indices_desc[i, j, torch.arange(all_scores.size(2)), target]


            # reshape final_tensor to [batch_size_query, num_layers, num_layers]
            final_tensor = final_tensor.permute(2, 0, 1)

            min_values, min_indices_flattened = torch.min(final_tensor.view(final_tensor.shape[0], -1), dim=1)

            # Calculate indices for 2D from the flattened indices
            min_row_indices = min_indices_flattened // final_tensor.shape[2]
            min_col_indices = min_indices_flattened % final_tensor.shape[2]

            # Pair them up
            min_indices = torch.stack((min_row_indices, min_col_indices), dim=1)
            # then target is



            #

            total_loss = self.compute_matryoshka_loss(full_layer_full_dim_scores,
                                                      target,
                                                      target_score)
            if num_layers > 1:
                layer_indices = random.sample(range(num_layers-1), 1)
            else:
                layer_indices = []

            for layer_idx in layer_indices:
                full_dim_scores = all_dim_scores[layer_idx, layer_idx, :, :]
                dot_product = full_dim_scores.sum(dim=-1)
                layer_loss = self.compute_matryoshka_loss(full_dim_scores, target, target_score)
                total_loss += layer_loss / (1 + layer_idx) / len(layer_indices)

                # last dim of full_dim_scores need to go through linear layer
                classifications = self.linear(full_dim_scores)

                classification_target = torch.zeros_like(classifications)
                # then for the score you get from each layer, if the maximum score from full_dim_scores is corresponding to the target, then target is 1, otherwise 0
                for i in range(classifications.shape[0]):
                    target_idx = target[i]
                    target_score = dot_product[i][target_idx]


                    last_layer_max_idx = torch.max(target_score[i])
                    last_layer_max_score = target_score[i][last_layer_max_idx]

                    # if max_idx >
                    #
                    #     classification_target[i] = 1
                print(classifications)
                print(classifications.shape)
                print(classification_target)
                print(classification_target.shape)
                exit(0)






            #####################################################################################

            # # next is dl_divergence loss
            # if self.kl_divergence_weight > 0:
            #     # Compute KL divergence loss





                # max_scores = torch.max(torch.stack(all_scores, dim=0), dim=0)[0]  # Get the max scores across layers
                #print("dim of max_scores: ", max_scores.shape)
                # layer_max_scores.append(max_scores)  # Append max scores of current query layer
            #print("dim of layer_max_scores: ", torch.stack(layer_max_scores, dim=0).shape)
            # final scores should be averaged on max_scores for query_doc
            # Average max scores across all query layers
            # final_scores = torch.mean(torch.stack(layer_max_scores, dim=0), dim=0)  # Average across the layers
            #print("dim of final_scores: ", final_scores.shape)
        else:
            raise NotImplementedError('Evaluation mode not implemented yet')

        return EncoderOutput(
            loss=total_loss,
            scores=target_score,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, is_trainable=True)
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False
                )
                lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                kl_divergence_weight=model_args.kl_divergence_weight,
                layer_list=model_args.layer_list.split(',') if model_args.layer_list else None,
                #embedding_dim_list=model_args.embedding_dim_list.split(',') if model_args.embedding_dim_list else None
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                kl_divergence_weight=model_args.kl_divergence_weight,
                layer_list=model_args.layer_list.split(',') if model_args.layer_list else None,
                #embedding_dim_list=model_args.embedding_dim_list.split(',') if model_args.embedding_dim_list else None
            )
        return model