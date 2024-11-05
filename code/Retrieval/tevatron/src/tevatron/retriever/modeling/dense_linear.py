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


class DenseRussianDollModellinear(DenseModel):

    def __init__(self,
                 kl_divergence_weight: float = 0.0,
                 layer_list: List = None,
                 embedding_dim_list: List = None,
                 linear_activation: bool = False,
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

        # create linear component, one for each layer and each dimention
        self.linear_layers = nn.ModuleDict()
        for layer in self.layer_list:
            for dim in self.embedding_dim_list:
                if linear_activation:
                    # load from the original
                    original_pooler = self.encoder.pooler.dense
                    original_pooler_data = original_pooler.weight.data.clone()[:, :dim]
                    original_pooler_bias = original_pooler.bias.data.clone()
                    print("original_pooler_data: ", original_pooler_data.shape)
                    self.linear_layers[f'layer_{layer}_dim_{dim}'] = nn.Sequential(
                        nn.Linear(original_pooler_data.size(0), original_pooler_data.size(1), bias=True),
                        nn.Tanh()
                    )
                    self.linear_layers[f'layer_{layer}_dim_{dim}'][0].weight.data = original_pooler_data
                    self.linear_layers[f'layer_{layer}_dim_{dim}'][0].bias.data = original_pooler_bias
                else:
                    self.linear_layers[f'layer_{layer}_dim_{dim}'] = nn.Linear(self.encoder.config.hidden_size, dim)

        self.layer_indices_map = {layer: i for i, layer in enumerate(self.layer_list)}
        self.dim_indices_map = {dim: i for i, dim in enumerate(self.embedding_dim_list)}

    def gradient_checkpointing_enable(self, **kwargs):
        # check if the encoder has gradient_checkpointing_enable method
        if hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()
        else:
            self.encoder.model.gradient_checkpointing_enable()


    def encode_query(self, qry):
        query_output = self.encoder(**qry, return_dict=True, output_hidden_states=True)
        all_hidden_states = torch.stack(query_output.hidden_states[1:])  # Stack all hidden states into a new dimension, except the first one which is the embedding layer
        reps = self._pooling(all_hidden_states, qry['attention_mask']) # Shape: [num_layers, num_layers, query_batch_size, passage_batch_size, hidden_dim]
        # for each layer and each dimention, apply linear transformation, then make final_reps become [num_layers, query_batch_size, passage_batch_size, dim, hidden_dim]
        reps = reps[self.layer_list]
        final_reps = torch.zeros((len(self.layer_list), qry['input_ids'].shape[0], len(self.embedding_dim_list), self.encoder.config.hidden_size)).to(reps.device)
        for layer_i, layer in enumerate(self.layer_list):
            for dim_i, dim in enumerate(self.embedding_dim_list):
                final_reps[layer_i, :, dim_i, :dim] = self.linear_layers[f'layer_{layer}_dim_{dim}'](reps[layer_i, :, :])
        return final_reps

    def encode_passage(self, psg):
        # Encode passage is the same as encode query
        return self.encode_query(psg)


    # def encode_query(self, qry, layer_num):
    #     query_output = self.encoder(**qry, return_dict=True, output_hidden_states=True)
    #     all_hidden_states = torch.stack(query_output.hidden_states[1:])  # Stack all hidden states into a new dimension, except the first one which is the embedding layer
    #     reps = self._pooling(all_hidden_states, qry['attention_mask']) # Shape: [num_layers, num_layers, query_batch_size, passage_batch_size, hidden_dim]
    #     # for each layer and each dimention, apply linear transformation, then make final_reps become [num_layers, query_batch_size, passage_batch_size, dim, hidden_dim]
    #     reps = reps[self.layer_list]
    #     final_reps = torch.zeros((len(self.layer_list), qry['input_ids'].shape[0], qry['input_ids'].shape[0], len(self.embedding_dim_list), self.encoder.config.hidden_size))
    #     for layer in range(len(self.layer_list)):
    #         for dim in self.embedding_dim_list:
    #             final_reps[layer, :, :, dim, :] = self.linear_layers[f'layer_{layer}_dim_{dim}'](reps[layer, :, :, :])
    #     return final_reps
    #
    # def encode_passage(self, psg):
    #     # Encode passage is the same as encode query
    #     return self.encode_query(psg)

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

    def compute_dl_divergence(self, scores, target):
        return F.kl_div(F.log_softmax(scores / self.temperature, dim=1), F.softmax(target / self.temperature, dim=1),
                        reduction='batchmean')

    def compute_matryoshka_loss(self, full_dim_scores, target, target_scores):
        total_loss = 0
        for dim_i, dim in enumerate(self.embedding_dim_list):
            scores = full_dim_scores[:, :, dim_i, :dim].sum(dim=-1)
            total_loss += self.compute_loss(scores / self.temperature, target)
            if self.kl_divergence_weight > 0:
                current_target_scores = target_scores[:, :, dim_i]
                kl_loss = self.compute_dl_divergence(scores, current_target_scores)
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
        # layer_max_scores = []
        # total_loss = 0 if self.training else None

        if self.training:
            # hard code for now
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            # Ensure that we are handling the dimensions correctly
            # Assume q_reps and p_reps are of shape [num_layers, batch_size, feature_dim]
            num_layers, batch_size, dimensions, embedding_dim = q_reps.shape
            ####################################################################################
            ############################# no for loop implementation #####################

            # first is to compute the similarity scores loss
            target = torch.arange(q_reps.size(1), device=q_reps.device, dtype=torch.long)
            target = target * (p_reps.size(1) // q_reps.size(1))
            all_dim_scores = torch.einsum('lqdh,Lpdh -> lLqpdh', q_reps, p_reps)  # [num_layers, num_layers, batch_size_query, batch_size_passage, dimention, hidden_dim]

            # last layer score for query_reps and passage_rep is target for dl_divergence
            full_layer_full_dim_scores = all_dim_scores[-1, -1, :, :, :]
            target_score = full_layer_full_dim_scores.sum(dim=-1).detach()

            total_loss = self.compute_matryoshka_loss(full_layer_full_dim_scores,
                                                      target,
                                                      target_score)
            if num_layers > 1:
                layer_indices = random.sample(range(num_layers-1), 1)
            else:
                layer_indices = []

            for layer_idx in layer_indices:
                full_dim_scores = all_dim_scores[layer_idx, layer_idx, :, :, :]
                layer_loss = self.compute_matryoshka_loss(full_dim_scores, target, target_score)
                total_loss += layer_loss / (1 + layer_idx) / len(layer_indices)

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
                embedding_dim_list=model_args.embedding_dim_list.split(',') if model_args.embedding_dim_list else None,
                linear_activation=model_args.linear_activation
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                kl_divergence_weight=model_args.kl_divergence_weight,
                layer_list=model_args.layer_list.split(',') if model_args.layer_list else None,
                embedding_dim_list=model_args.embedding_dim_list.split(',') if model_args.embedding_dim_list else None,
                linear_activation=model_args.linear_activation
            )

        return model



    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             layer_list: List = None,
             embedding_dim_list: List = None,
                linear_activation: bool = False,
             **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if lora_name_or_path:
            linear_layer_path = os.path.join(lora_name_or_path, 'linear_layer', 'linear_layers.pt')
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=pooling,
                normalize=normalize,
                layer_list=layer_list.split(',') if layer_list else None,
                embedding_dim_list=embedding_dim_list.split(',') if embedding_dim_list else None,
                linear_activation=linear_activation
            )
        else:
            linear_layer_path = os.path.join(model_name_or_path, 'linear_layer', 'linear_layers.pt')
            model = cls(
                encoder=base_model,
                pooling=pooling,
                normalize=normalize,
                layer_list=layer_list.split(',') if layer_list else None,
                embedding_dim_list=embedding_dim_list.split(',') if embedding_dim_list else None,
                linear_activation=linear_activation
            )
        with open(linear_layer_path, 'rb') as f:
            linear_layers_dict = torch.load(f)
            # Adjust the keys by stripping the prefix 'linear_layers.'
            model.linear_layers.load_state_dict(linear_layers_dict)
        return model

    def save(self, output_dir: str):
        # save also the linear layer with the model
        model_to_save = self.encoder.module if hasattr(self.encoder, 'module') else self.encoder
        model_to_save.save_pretrained(output_dir)
        self.linear_layers.save_pretrained(output_dir)



