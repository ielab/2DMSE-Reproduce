import os
from typing import Optional

import torch

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist
from .modeling import EncoderModel
from .modeling import DenseSeperatedModel


import logging
logger = logging.getLogger(__name__)


class TevatronTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(TevatronTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel, DenseSeperatedModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            if isinstance(self.model, DenseSeperatedModel):
                query_prefix = "query_encoder."
                passage_prefix = "passage_encoder."
                #should start either with query or passage

                assert all(k.startswith(query_prefix) or k.startswith(passage_prefix) for k in state_dict.keys()), list(state_dict.keys())
                query_state_dict = {k[len(query_prefix):]: v for k, v in state_dict.items() if k.startswith(query_prefix)}
                passage_state_dict = {k[len(passage_prefix):]: v for k, v in state_dict.items() if k.startswith(passage_prefix)}
                query_encoder_path = os.path.join(output_dir, "query_encoder")
                passage_encoder_path = os.path.join(output_dir, "passage_encoder")

                self.model.query_encoder.save_pretrained(
                    query_encoder_path, state_dict=query_state_dict, safe_serialization=self.args.save_safetensors
                )
                self.model.passage_encoder.save_pretrained(
                    passage_encoder_path, state_dict=passage_state_dict, safe_serialization=self.args.save_safetensors
                )

            else:
                prefix = 'encoder.'
                assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
                state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
                self.model.encoder.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(self, model, inputs):
        query, passage = inputs
        return model(query=query, passage=passage).loss

    def training_step(self, *args):
        return super(TevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor

