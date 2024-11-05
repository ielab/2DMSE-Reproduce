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




class DenseSeperatedSharedModel(DenseModel):

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 query_encoder_num_layers: int = 0,
                 passage_encoder_num_layers: int = 0,
                 dim: int = None,
                 ):
        super().__init__(
            encoder=encoder,
            pooling=pooling,
            normalize=normalize,
            temperature=temperature,
        )
        # if num_layers is -1, use the full size encoder

        self.query_encoder_num_layers = query_encoder_num_layers
        self.passage_encoder_num_layers = passage_encoder_num_layers
        self.dim = dim if dim is not None else encoder.config.hidden_size


    def encode_query(self, qry):
        query_hidden_states = self.encoder(**qry, return_dict=True, output_hidden_states=True)
        if self.query_encoder_num_layers > 0:
            query_hidden_states = query_hidden_states.hidden_states[self.query_encoder_num_layers]
        else:
            query_hidden_states = query_hidden_states.hidden_states[-1]
        return self._pooling(query_hidden_states, qry['attention_mask'])

    def encode_passage(self, psg):
        passage_hidden_states = self.encoder(**psg, return_dict=True, output_hidden_states=True)
        if self.passage_encoder_num_layers > 0:
            passage_hidden_states = passage_hidden_states.hidden_states[self.passage_encoder_num_layers]
        else:
            passage_hidden_states = passage_hidden_states.hidden_states[-1]
        return self._pooling(passage_hidden_states, psg['attention_mask'])

    def gradient_checkpointing_enable(self, **kwargs):
        # check if the encoder has gradient_checkpointing_enable method
        if hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()
        else:
            self.encoder.model.gradient_checkpointing_enable()

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        reps = reps[:, :self.dim]
        return reps

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
                query_encoder_num_layers=model_args.query_encoder_num_layers,
                passage_encoder_num_layers=model_args.passage_encoder_num_layers,
                dim=model_args.dim
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                query_encoder_num_layers=model_args.query_encoder_num_layers,
                passage_encoder_num_layers=model_args.passage_encoder_num_layers,
                dim=model_args.dim
            )
        return model


    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             query_encoder_num_layers: int = 0,
             passage_encoder_num_layers: int = 0,
             **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=pooling,
                normalize=normalize,
                query_encoder_num_layers=query_encoder_num_layers,
                passage_encoder_num_layers=passage_encoder_num_layers,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=pooling,
                normalize=normalize,
                query_encoder_num_layers=query_encoder_num_layers,
                passage_encoder_num_layers=passage_encoder_num_layers,
            )
        return model


class DenseRussianDollModel(DenseModel):

    def __init__(self,
                 kl_divergence_weight: float = 0.0,
                 layer_list: List = None,
                 embedding_dim_list: List = None,
                 representation_kl: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.kl_divergence_weight = kl_divergence_weight
        if layer_list is not None:
            self.layer_list = [int(i) - 1 for i in layer_list]
        else:
            self.layer_list = [-1]

        if embedding_dim_list is not None:
            self.embedding_dim_list = [int(i) for i in embedding_dim_list]
        else:
            self.embedding_dim_list = [-1]
        self.representation_kl = representation_kl


    def gradient_checkpointing_enable(self, **kwargs):
        # check if the encoder has gradient_checkpointing_enable method
        if hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()
        else:
            self.encoder.model.gradient_checkpointing_enable()


    def encode_query(self, qry):
        query_output = self.encoder(**qry, return_dict=True, output_hidden_states=True)
        all_hidden_states = torch.stack(query_output.hidden_states[1:])  # Stack all hidden states into a new dimension, except the first one which is the embedding layer
        return self._pooling(all_hidden_states, qry['attention_mask'])

    def encode_passage(self, psg):
        # Encode passage is the same as encode query
        return self.encode_query(psg)

    def _pooling(self, all_hidden_states, attention_mask):
        # all_hidden_states shape: [num_layers, batch_size, seq_length, hidden_dim]
        if self.pooling in ['cls', 'first']:
            # Select the first token for all layers
            reps = all_hidden_states[:, :, 0, :]  # Shape: [num_layers, batch_size, hidden_dim]
        elif self.pooling in ['mean', 'avg', 'average']:
            expanded_mask = attention_mask[None, ... , None]
            masked_hiddens = all_hidden_states.masked_fill(~expanded_mask.bool(), 0.0)
            reps = masked_hiddens.sum(dim=2) / attention_mask.sum(dim=1)[..., None]
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

    def compute_dl_divergence_embedding(self, embedding, target_embedding):
        loss = self.kl_divergence_weight * F.kl_div(F.log_softmax(embedding / self.temperature, dim=1),
                                                    F.softmax(target_embedding / self.temperature, dim=1),
                                                    reduction='batchmean')
        return loss
    # def compute_dl_divergence_batch(self, full_dim_scores, target_embeds):
    #     total_loss = 0
    #     for dim in self.embedding_dim_list:
    #         scores = full_dim_scores[:, :, :dim].sum(dim=-1)
    #         # for dimentional different to target, use pca to project to the same dimention, otherwise, use the original dimention
    #         # if dim != target_embed.shape[-1]:
    #         #
    #         #     scores = self.pca_reduce(scores, target_embed.shape[-1])
    #         # total_loss += F.kl_div(F.log_softmax(scores / self.temperature, dim=1), F.softmax(target_scores / self.temperature, dim=1), reduction='batchmean')
    #     return total_loss

    def compute_matryoshka_loss(self, full_dim_scores, target, target_scores):
        total_loss = 0
        for dim in self.embedding_dim_list:
            scores = full_dim_scores[:, :, :dim].sum(dim=-1)
            total_loss += self.compute_loss(scores / self.temperature, target)
            if self.kl_divergence_weight > 0:
                kl_loss = self.compute_dl_divergence(scores, target_scores)
                total_loss += self.kl_divergence_weight * kl_loss
        return total_loss

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None
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

            ############################# Double for loop implementation #####################
            # Compute target indices for this batch
            # target = torch.arange(q_reps.size(1), device=q_reps.device, dtype=torch.long)
            # target = target * (p_reps.size(1) // q_reps.size(1))
            # Process each layer individually
            # for layer_idx_q in range(num_layers):
            #     # all_scores = []
            #     layer_q_reps = q_reps[layer_idx_q]
            #     weight_q = 1 / (math.log(layer_idx_q + 1) + 1) if layer_idx_q != num_layers - 1 else 1
            #     # Weight for query layer
            #     for layer_idx_p in range(num_layers):
            #         layer_p_reps = p_reps[layer_idx_p]
            #         weight_p = 1 / (math.log(layer_idx_p + 1) + 1) if layer_idx_p != num_layers - 1 else 1
            #
            #         # Compute similarity scores for the current layer
            #         scores = self.compute_similarity(layer_q_reps, layer_p_reps)
            #         # all_scores.append(scores)
            #         # Calculate loss for this layer
            #         loss = self.compute_loss(scores / self.temperature, target) * weight_q * weight_p
            #         if self.is_ddp:
            #             loss *= self.world_size  # Scale loss according to the number of GPUs
            #
            #         total_loss += loss
            ####################################################################################
            ############################# no for loop implementation #####################

            # first is to compute the similarity scores loss
            target = torch.arange(q_reps.size(1), device=q_reps.device, dtype=torch.long)
            target = target * (p_reps.size(1) // q_reps.size(1))
            all_dim_scores = torch.einsum('lqh,Ldh -> lLqdh', q_reps, p_reps)  # [num_layers, num_layers, batch_size, hidden_dim]

            # last layer score for query_reps and passage_rep is target for dl_divergence
            full_layer_full_dim_scores = all_dim_scores[-1, -1, :, :]
            target_score = full_layer_full_dim_scores.sum(dim=-1).detach()

            q_reps_final = q_reps[-1]
            p_reps_final = p_reps[-1]

            total_loss = self.compute_matryoshka_loss(full_layer_full_dim_scores,
                                                      target,
                                                      target_score)
            if num_layers > 1:
                layer_indices = random.sample(range(num_layers-1), 1)
            else:
                layer_indices = []

            for layer_idx in layer_indices:
                full_dim_scores = all_dim_scores[layer_idx, layer_idx, :, :]
                layer_loss = self.compute_matryoshka_loss(full_dim_scores, target, target_score)
                total_loss += layer_loss / (1 + layer_idx) / len(layer_indices)
                query_reps = q_reps[layer_idx]
                passage_reps = p_reps[layer_idx]
                # kl embedding loss
                if self.representation_kl:
                    kl_embedding_loss = 0
                    kl_embedding_loss  += self.compute_dl_divergence_embedding(query_reps, q_reps_final) / (1 + layer_idx) / len(layer_indices)
                    kl_embedding_loss  += self.compute_dl_divergence_embedding(passage_reps, p_reps_final) / (1 + layer_idx) / len(layer_indices)
                    print("kl_embedding_loss: ", kl_embedding_loss)
                    total_loss += kl_embedding_loss

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
                representation_kl=model_args.representation_kl,
                layer_list=model_args.layer_list.split(',') if model_args.layer_list else None,
                embedding_dim_list=model_args.embedding_dim_list.split(',') if model_args.embedding_dim_list else None
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                kl_divergence_weight=model_args.kl_divergence_weight,
                representation_kl=model_args.representation_kl,
                layer_list=model_args.layer_list.split(',') if model_args.layer_list else None,
                embedding_dim_list=model_args.embedding_dim_list.split(',') if model_args.embedding_dim_list else None
            )
        return model







class Dense2DRussianDollModel(DenseRussianDollModel):

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None
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

            target = torch.arange(q_reps.size(1), device=q_reps.device, dtype=torch.long)
            target = target * (p_reps.size(1) // q_reps.size(1))
            all_dim_scores = torch.einsum('lqh,Ldh -> lLqdh', q_reps, p_reps)  # [num_layers, num_layers, batch_size, num_d, hidden_dim]

            # last layer score for query_reps and passage_rep is target for dl_divergence
            full_layer_full_dim_scores = all_dim_scores[-1, -1, :, :, :]
            full_layer_full_dim_reps = torch.cat([q_reps[-1], p_reps[-1]], dim=0)

            # full layer full dim loss
            loss = self.compute_loss(full_layer_full_dim_scores.sum(dim=-1) / self.temperature, target)

            layer_ind = random.choice(range(num_layers-1)) if len(self.layer_list) > 1 else None
            dim_ind = random.choice(range(len(self.embedding_dim_list)-1)) if len(self.embedding_dim_list) > 1 else None

            if layer_ind is None and dim_ind is None:
                return loss

            # sampled layer full dim loss
            if layer_ind is not None:
                layer_full_dim_scores = all_dim_scores[layer_ind, layer_ind, :, :, :]
                layer_loss = self.compute_loss(layer_full_dim_scores.sum(dim=-1) / self.temperature, target)
                loss += layer_loss / (1 + layer_ind)

                layer_full_dim_reps = torch.cat([q_reps[layer_ind], p_reps[layer_ind]], dim=0)
                # kl loss
                kl_div_loss = F.kl_div(F.log_softmax(layer_full_dim_reps / 0.3, dim=1),
                                       F.softmax(full_layer_full_dim_reps / 0.3, dim=1),
                                       reduction='batchmean')
                loss += kl_div_loss

            # sampled dim full layer loss
            if dim_ind is not None:
                full_layer_dim_scores = full_layer_full_dim_scores[:, :, :self.embedding_dim_list[dim_ind]]
                dim_loss = self.compute_loss(full_layer_dim_scores.sum(dim=-1) / self.temperature, target)
                loss += dim_loss

            # sampled layer sampled dim loss
            if layer_ind is not None and dim_ind is not None:
                layer_dim_scores = all_dim_scores[layer_ind, layer_ind, :, :, :self.embedding_dim_list[dim_ind]]
                layer_dim_loss = self.compute_loss(layer_dim_scores.sum(dim=-1) / self.temperature, target)
                loss += layer_dim_loss / (1 + layer_ind)

                layer_dim_reps = torch.cat([q_reps[layer_ind][:, :self.embedding_dim_list[dim_ind]],
                                            p_reps[layer_ind][:, :self.embedding_dim_list[dim_ind]]], dim=0)
                # kl loss
                kl_div_loss = F.kl_div(F.log_softmax(layer_dim_reps / 0.3, dim=1),
                                        F.softmax(full_layer_full_dim_reps[:, :self.embedding_dim_list[dim_ind]] / 0.3,
                                                  dim=1),
                                        reduction='batchmean')
                loss += kl_div_loss
        else:
            raise NotImplementedError('Evaluation mode not implemented yet')

        return EncoderOutput(
            loss=loss,
            scores=None,
            q_reps=q_reps,
            p_reps=p_reps,
        )




class DenseRussianDollModelSeperateLayer(DenseRussianDollModel):

    def __init__(self,
                 kl_divergence_weight: float = 0.0,
                 layer_list_query: List = None,
                 layer_list_passage: List = None,
                 embedding_dim_list: List = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.kl_divergence_weight = kl_divergence_weight
        if layer_list_query is not None:
            self.layer_list_query = [int(i) - 1 for i in layer_list_query]
        else:
            self.layer_list_query = [-1]

        if layer_list_passage is not None:
            self.layer_list_passage = [int(i) - 1 for i in layer_list_passage]
        else:
            self.layer_list_passage = [-1]


        if embedding_dim_list is not None:
            self.embedding_dim_list = [int(i) for i in embedding_dim_list]
        else:
            self.embedding_dim_list = [-1]



    def compute_matryoshka_loss(self, full_dim_scores, target, target_scores):
        total_loss = 0
        for dim in self.embedding_dim_list:
            scores = full_dim_scores[:, :, :dim].sum(dim=-1)
            total_loss += self.compute_loss(scores / self.temperature, target)
            if self.kl_divergence_weight > 0:
                kl_loss = self.compute_dl_divergence(scores, target_scores)
                total_loss += self.kl_divergence_weight * kl_loss
        return total_loss

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None
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
            q_reps = q_reps[self.layer_list_query]
            p_reps = p_reps[self.layer_list_passage]

            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            # Ensure that we are handling the dimensions correctly
            # Assume q_reps and p_reps are of shape [num_layers, batch_size, feature_dim]
            num_layers_query, batch_size, embedding_dim = q_reps.shape
            num_layers_passage, batch_size, embedding_dim = p_reps.shape

            ############################# Double for loop implementation #####################
            # Compute target indices for this batch
            # target = torch.arange(q_reps.size(1), device=q_reps.device, dtype=torch.long)
            # target = target * (p_reps.size(1) // q_reps.size(1))
            # Process each layer individually
            # for layer_idx_q in range(num_layers):
            #     # all_scores = []
            #     layer_q_reps = q_reps[layer_idx_q]
            #     weight_q = 1 / (math.log(layer_idx_q + 1) + 1) if layer_idx_q != num_layers - 1 else 1
            #     # Weight for query layer
            #     for layer_idx_p in range(num_layers):
            #         layer_p_reps = p_reps[layer_idx_p]
            #         weight_p = 1 / (math.log(layer_idx_p + 1) + 1) if layer_idx_p != num_layers - 1 else 1
            #
            #         # Compute similarity scores for the current layer
            #         scores = self.compute_similarity(layer_q_reps, layer_p_reps)
            #         # all_scores.append(scores)
            #         # Calculate loss for this layer
            #         loss = self.compute_loss(scores / self.temperature, target) * weight_q * weight_p
            #         if self.is_ddp:
            #             loss *= self.world_size  # Scale loss according to the number of GPUs
            #
            #         total_loss += loss
            ####################################################################################
            ############################# no for loop implementation #####################

            # first is to compute the similarity scores loss
            target = torch.arange(q_reps.size(1), device=q_reps.device, dtype=torch.long)
            target = target * (p_reps.size(1) // q_reps.size(1))
            all_dim_scores = torch.einsum('lqh,Ldh -> lLqdh', q_reps, p_reps)  # [num_layers, num_layers, batch_size, hidden_dim]

            # last layer score for query_reps and passage_rep is target for dl_divergence
            full_layer_full_dim_scores = all_dim_scores[-1, -1, :, :]
            target_score = full_layer_full_dim_scores.sum(dim=-1).detach()

            total_loss = self.compute_matryoshka_loss(full_layer_full_dim_scores,
                                                      target,
                                                      target_score)
            if num_layers_query > 1:
                layer_indices_query = random.sample(range(num_layers_query-1), 1)
            else:
                layer_indices_query = []

            if num_layers_passage > 1:
                layer_indices_passage = random.sample(range(num_layers_passage-1), 1)
            else:
                layer_indices_passage = []

            for layer_idx_q in layer_indices_query:
                for layer_idx_p in layer_indices_passage:
                    full_dim_scores = all_dim_scores[layer_idx_q, layer_idx_p, :, :]
                    layer_loss = self.compute_matryoshka_loss(full_dim_scores, target, target_score)
                    total_loss += layer_loss / (1 + layer_idx_q + layer_idx_p) / ((len(layer_indices_query) + len(layer_indices_passage)))
        else:
            raise NotImplementedError('Evaluation mode not implemented yet')


        return EncoderOutput(
            loss=total_loss,
            scores=target_score,
            q_reps=q_reps,
            p_reps=p_reps,
        )




class DenseSeperatedModel(EncoderModel):

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 query_encoder_num_layers: int = 0,
                 passage_encoder_num_layers: int = 0,
                 ):
        super().__init__(
            encoder=encoder,
            pooling=pooling,
            normalize=normalize,
            temperature=temperature,
        )
        # if num_layers is 0, use the original encoder; otherwise only select the first num_layers


        self.query_encoder = copy.deepcopy(self.encoder)

        if query_encoder_num_layers > 0:
            self.query_encoder.layers = self.query_encoder.layers[:query_encoder_num_layers]


        self.passage_encoder = copy.deepcopy(self.encoder)

        if passage_encoder_num_layers > 0:
            self.passage_encoder.layers = self.passage_encoder.layers[:passage_encoder_num_layers]

        del self.encoder


    def encode_query(self, qry):
        query_hidden_states = self.query_encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        return self._pooling(query_hidden_states, qry['attention_mask'])

    def encode_passage(self, psg):
        # encode passage is the same as encode query
        passage_hidden_states = self.passage_encoder(**psg, return_dict=True)
        passage_hidden_states = passage_hidden_states.last_hidden_state
        return self._pooling(passage_hidden_states, psg['attention_mask'])

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

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

        model = cls(
            encoder=base_model,
            pooling=model_args.pooling,
            normalize=model_args.normalize,
            temperature=model_args.temperature,
            query_encoder_num_layers=model_args.query_encoder_num_layers,
            passage_encoder_num_layers=model_args.passage_encoder_num_layers
        )

        query_encoder = model.query_encoder
        passage_encoder = model.passage_encoder

        if train_args.gradient_checkpointing:
            query_encoder.gradient_checkpointing()
            passage_encoder.gradient_checkpointing()

        if model_args.lora or model_args.lora_name_or_path:
            if model_args.lora_name_or_path:
                lora_model_query_encoder = PeftModel.from_pretrained(query_encoder, model_args.lora_name_or_path, is_trainable=True)
                lora_model_passage_encoder = PeftModel.from_pretrained(passage_encoder, model_args.lora_name_or_path, is_trainable=True)
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
                lora_model_query_encoder = get_peft_model(query_encoder, lora_config)
                lora_model_passage_encoder = get_peft_model(passage_encoder, lora_config)

            model.query_encoder = lora_model_query_encoder
            model.passage_encoder = lora_model_passage_encoder

        return model

    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             query_encoder_num_layers: int = 0,
             passage_encoder_num_layers: int = 0,

             **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0

        model = cls(
            encoder=base_model,
            pooling=pooling,
            normalize=normalize,
            query_encoder_num_layers=query_encoder_num_layers,
            passage_encoder_num_layers=passage_encoder_num_layers
        )

        if lora_name_or_path:
            query_encoder_path = os.path.join(lora_name_or_path, 'query_encoder')
            passage_encoder_path = os.path.join(lora_name_or_path, 'passage_encoder')
            lora_config_query = LoraConfig.from_pretrained(query_encoder_path, **hf_kwargs)
            lora_config_passage = LoraConfig.from_pretrained(passage_encoder_path, **hf_kwargs)
            lora_model_query_encoder = PeftModel.from_pretrained(model.query_encoder, query_encoder_path, config=lora_config_query)
            lora_model_passage_encoder = PeftModel.from_pretrained(model.passage_encoder, passage_encoder_path, config=lora_config_passage)
            lora_model_query_encoder = lora_model_query_encoder.merge_and_unload()
            lora_model_passage_encoder = lora_model_passage_encoder.merge_and_unload()
            model.query_encoder = lora_model_query_encoder
            model.passage_encoder = lora_model_passage_encoder

        return model

    def save(self, output_dir: str):
        # save both query and passage encoder
        query_encoder_path = os.path.join(output_dir, "query_encoder")
        passage_encoder_path = os.path.join(output_dir, "passage_encoder")

        self.query_encoder.save_pretrained(query_encoder_path)
        self.passage_encoder.save_pretrained(passage_encoder_path)

    def gradient_checkpointing_enable(self, **kwargs):
        self.query_encoder.gradient_checkpointing_enable()
        self.passage_encoder.gradient_checkpointing_enable()
