import torch
import logging
from .encoder import EncoderModel, EncoderOutput
from .dense import DenseModel
from transformers import PreTrainedModel, AutoModel
from typing import Dict, Optional
import copy
from tevatron.retriever.arguments import ModelArguments, DataArguments, Matryoshka2dDenseModelArguments, \
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


class Matryoshka2dDenseModelV2(DenseModel):

    def __init__(self,
                 kl_divergence_weight: float = 0.0,
                 layer_list: List = None,
                 embedding_dim_list: List = None,
                 # sub_model_sampling: bool = True,
                 #sub_model_sampling: bool = False,
                 score_align: bool = False,
                 embedding_align: bool = False,
                 compare_to_last_layer: bool = False,
                 passage_to_last_layer: bool = False,
                 sub_layer_full_dim: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.kl_divergence_weight = kl_divergence_weight


        if embedding_dim_list is not None:
            self.target_embedding_dims = [int(i) for i in embedding_dim_list]
        else:
            self.target_embedding_dims = [128]

        if layer_list is not None:
            self.layer_list = [int(i) for i in layer_list]
        else:
            #layer list is all the layers
            self.layer_list = None
        self.score_align = score_align
        self.embedding_align = embedding_align
        self.compare_to_last_layer = compare_to_last_layer
        self.sub_layer_full_dim = sub_layer_full_dim
        self.passage_to_last_layer = passage_to_last_layer
        print(f"Layer list: {self.layer_list}")
        print(f"Target embedding dim: {self.target_embedding_dims}")
        print(f"Score align: {self.score_align}")
        print(f"Compare to last layer: {self.compare_to_last_layer}")
        print(f"Sub layer full dim: {self.sub_layer_full_dim}")
        print(f"Passage to last layer: {self.passage_to_last_layer}")
        #print(f"Add full sub model sampling: {self.sub_model_sampling}")

        if self.compare_to_last_layer and not self.score_align:
            raise ValueError("compare_to_last_layer can only be set to True if score_align is also set to True")

        #self.embedding_dim_list = embedding_dim_list

    # def pca_reduce(self, embeddings, target_dim):
    #     """Get top-k features via PCA.
    #
    #     :param embeddings: torch.Tensor. Input tensor. Shape: [batch_size, hidden_dim]
    #     :param target_dim: int. Target dimension (k) for top-k features.
    #
    #     :return: torch.Tensor. Top-k features. Shape: [batch_size, target_dim], each embedding is reduced to target_dim.
    #     """

    @torch.no_grad()
    def pca_reduce(self, m: torch.Tensor, k: int) -> torch.Tensor:
        """ Get topk feature via PCA.

        :param m: torch.Tensor. Input tensor.
        :param k: int. Top-k feature size.

        :return: torch.Tensor. Top-k feature.
        """
        A = F.softmax(m.T @ m / m.shape[-1] ** 0.5, dim=-1)
        u, s, _ = torch.svd_lowrank(A, q=k)
        # top-k principal components
        topk_deps = u @ torch.diag(s)
        return m @ topk_deps


    @torch.no_grad()
    def pca_reduces(self, m: torch.Tensor, ks: list) -> list:
        """ Get topk feature via PCA.

        :param m: torch.Tensor. Input tensor.
        :param ks: list. Top-k feature size.

        :return: torch.Tensor. Top-k feature.
        """
        A = F.softmax(m.T @ m / m.shape[-1] ** 0.5, dim=-1)
        u, s, _ = torch.svd_lowrank(A, q=max(ks))
        # top-k principal components
        topk_deps = u @ torch.diag(s)
        return [m @ topk_deps[:, :k] for k in ks]

    def gradient_checkpointing_enable(self, **kwargs):
        # check if the encoder has gradient_checkpointing_enable method
        if hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()
        else:
            self.encoder.model.gradient_checkpointing_enable()

    def encode_query(self, qry):
        query_output = self.encoder(**qry, return_dict=True, output_hidden_states=True)
        all_hidden_states = torch.stack(query_output.hidden_states)  # Note the layer 0 is embedding layer
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

    def compute_alignment_loss(self, reps, target_dim, reduced_reps=None):
        # compute loss using combined kl_divergence and mse loss, from pca reduced embeddings vs top k dims
        # reps shape: [batch_size, hidden_dim]
        # target_dim: int
        batch_size, hidden_dim = reps.shape
        if reduced_reps is None:
            reduced_reps = self.pca_reduce(reps, target_dim)

        truncated_reps = reps[:, :target_dim]
        mse_loss = F.mse_loss(truncated_reps, reduced_reps)
        kl_loss = F.kl_div(F.log_softmax(truncated_reps/ self.temperature, dim=1), F.softmax(reduced_reps/ self.temperature, dim=1), reduction='batchmean')
        return (mse_loss + kl_loss)/batch_size




    def compute_score_loss(self, layer_scores, target_scores):
        # compute loss using dot product scores
        batch_size, num_docs = layer_scores.shape
        mse_loss = F.mse_loss(layer_scores, target_scores)
        kl_loss = F.kl_div(F.log_softmax(layer_scores/ self.temperature, dim=1), F.softmax(target_scores/ self.temperature, dim=1), reduction='batchmean')
        total_loss = (mse_loss + kl_loss)/batch_size/num_docs
        return total_loss



    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        #total_loss = 0 if self.training else None

        if self.training:
            if self.layer_list is not None:
                q_reps = q_reps[self.layer_list]
                p_reps = p_reps[self.layer_list]
            else:
                q_reps = q_reps[1:]
                p_reps = p_reps[1:]
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            # Assume q_reps and p_reps are of shape [num_layers, batch_size, hidden_size]
            num_layers, batch_size, embedding_dim = q_reps.shape
            target = torch.arange(q_reps.size(1), device=q_reps.device, dtype=torch.long)
            target = target * (p_reps.size(1) // q_reps.size(1))

            # first is to compute the element-wise scores for each layer and full dim
            if self.passage_to_last_layer:
                p_reps = p_reps[-1]
                all_scores = torch.einsum('lqh,dh -> lqdh', q_reps, p_reps) # [num_layers, batch_size (num_queries), num_docs, hidden_size]
            else:
                all_scores = torch.einsum('lqh,ldh -> lqdh', q_reps, p_reps) # [num_layers, batch_size (num_queries), num_docs, hidden_size]

            # full layer full dim predicted scores and loss
            # first is the last layer losses
            last_layer_index = num_layers - 1
            full_layer_full_dim_scores = all_scores[last_layer_index, :, :]
            full_layer_scores = full_layer_full_dim_scores.sum(dim=-1)  # [batch_size, num_docs], dot product scores
            total_loss = self.compute_loss(full_layer_scores / self.temperature, target)


            for target_dim in self.target_embedding_dims:
                layer_dim_scores = all_scores[last_layer_index, :, :,
                                   :target_dim]  # [batch_size, num_docs, dim]
                layer_scores = layer_dim_scores.sum(dim=-1)  # dot product
                total_loss += self.compute_loss(layer_scores / self.temperature, target)
                if self.score_align:
                    # compute score based loss
                    # for target dim, just compute score based loss
                    full_layer_scores = full_layer_scores.detach()
                    total_loss += self.compute_score_loss(layer_scores, full_layer_scores)

            if self.embedding_align:
                if self.compare_to_last_layer:
                    q_reps_final = q_reps[-1]
                    p_reps_final = p_reps[-1]
                    for target_dim in self.target_embedding_dims:
                        total_loss += self.compute_alignment_loss(q_reps_final, target_dim)
                        total_loss += self.compute_alignment_loss(p_reps_final, target_dim)
                    q_reps_final_reduced = self.pca_reduces(q_reps_final, self.target_embedding_dim).detach()
                    p_reps_final_reduced = self.pca_reduces(p_reps_final, self.target_embedding_dim).detach()

                else:
                    q_reps_final = q_reps[-1]
                    if not self.passage_to_last_layer:
                        p_reps_final = p_reps[-1]
                    else:
                        p_reps_final = p_reps
                    for target_dim in self.target_embedding_dims:
                        total_loss += self.compute_alignment_loss(q_reps_final, target_dim)
                        total_loss += self.compute_alignment_loss(p_reps_final, target_dim)


            for layer_idx in range(num_layers):
                if layer_idx == last_layer_index:
                    # full layer full dim loss is already computed
                    continue
                # \dfrac{1}{1 + \ln(i)} & \text{for } i < L \\
                layer_weight = 1 / (1 + math.log(layer_idx + 1))

                for target_dim in self.target_embedding_dims:
                    current_layer_dim_scores = all_scores[layer_idx, :, :,
                                               :target_dim]  # [batch_size, num_docs, dim]
                    current_layer_scores = current_layer_dim_scores.sum(dim=-1)  # dot product
                    total_loss += layer_weight * self.compute_loss(current_layer_scores / self.temperature, target)
                    if self.score_align:
                        # compute score based loss
                        # for target dim, just compute score based loss
                        if self.compare_to_last_layer:
                            total_loss += layer_weight * self.compute_score_loss(current_layer_scores,
                                                                                 full_layer_scores)
                        else:
                            # current_layer_full_scores = all_scores[layer_idx, :, :]
                            # target_scores = current_layer_full_scores.sum(dim=-1).detach()
                            if not self.sub_layer_full_dim:
                                current_layer_full_scores = all_scores[layer_idx, :, :]
                                current_layer_full_scores = current_layer_full_scores.sum(dim=-1)  # dot product
                            # total_loss += layer_weight * self.compute_loss(current_layer_full_scores / self.temperature, target)
                            target_scores = current_layer_full_scores.detach()
                            total_loss += layer_weight * self.compute_score_loss(current_layer_scores, target_scores)

                if self.sub_layer_full_dim:
                    # compute loss using dot product scores
                    current_layer_full_scores = all_scores[layer_idx, :, :]
                    current_layer_full_scores = current_layer_full_scores.sum(dim=-1)
                    total_loss += layer_weight * self.compute_loss(current_layer_full_scores / self.temperature, target)

                if self.embedding_align:
                    if self.compare_to_last_layer:
                        for target_idx, target_dim in enumerate(self.target_embedding_dims):
                            total_loss += layer_weight * self.compute_alignment_loss(q_reps[layer_idx], target_dim, q_reps_final_reduced[target_idx])
                            total_loss += layer_weight * self.compute_alignment_loss(p_reps[layer_idx], target_dim, p_reps_final_reduced[target_idx])
                    else:
                        q_reps_reduced = self.pca_reduces(q_reps[layer_idx], self.target_embedding_dims)
                        for target_idx, target_dim in enumerate(self.target_embedding_dims):
                            total_loss += layer_weight * self.compute_alignment_loss(q_reps[layer_idx], target_dim, q_reps_reduced[target_idx])
                        if not self.passage_to_last_layer:
                            p_reps_reduced = self.pca_reduces(p_reps[layer_idx], self.target_embedding_dims)
                            for target_idx, target_dim in enumerate(self.target_embedding_dims):
                                total_loss += layer_weight * self.compute_alignment_loss(p_reps[layer_idx], target_dim, p_reps_reduced[target_idx])
        else:
            raise NotImplementedError('Evaluation mode not implemented yet')

        return EncoderOutput(
            loss=total_loss,
            scores=full_layer_scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    @classmethod
    def build(
            cls,
            model_args: Matryoshka2dDenseModelArguments,
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
                sub_model_sampling=model_args.sub_model_sampling,
                score_align=model_args.score_align,
                compare_to_last_layer=model_args.compare_to_last_layer,
                passage_to_last_layer=model_args.passage_to_last_layer,
                sub_layer_full_dim=model_args.sub_layer_full_dim,
                embedding_align=model_args.embedding_align,

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
                sub_model_sampling=model_args.sub_model_sampling,
                score_align=model_args.score_align,
                compare_to_last_layer=model_args.compare_to_last_layer,
                passage_to_last_layer=model_args.passage_to_last_layer,
                sub_layer_full_dim=model_args.sub_layer_full_dim,
                embedding_align=model_args.embedding_align,
            )
        return model

