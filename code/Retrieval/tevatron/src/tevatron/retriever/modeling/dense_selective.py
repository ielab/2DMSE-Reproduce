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


class DenseSelectiveModel(DenseModel):
    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 layer_to_train: int = 0,
                 ):
        super().__init__(
            encoder=encoder,
            pooling=pooling,
            normalize=normalize,
            temperature=temperature,
        )
        self.layer_to_train = layer_to_train


    def encode_query(self, qry):
        query_hidden_states = self.encoder(**qry, return_dict=True, output_hidden_states=True)
        if self.layer_to_train > 0:
            query_hidden_states = query_hidden_states.hidden_states[self.layer_to_train]
        else:
            query_hidden_states = query_hidden_states.hidden_states[-1]
        return self._pooling(query_hidden_states, qry['attention_mask'])

    def encode_passage(self, psg):
        return self.encode_query(psg)

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
                layer_to_train=model_args.layer_to_train
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                layer_to_train=model_args.layer_to_train
            )
        return model
