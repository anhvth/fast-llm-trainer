import transformers
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math
import torch
import time
import os
import json
from dataclasses import dataclass, field
from typing import *
import numpy as np
import random
import torch
from tqdm import tqdm
import os.path as osp
from copy import deepcopy
# from modeling.dynamic_batching_trainer import split_then_stack
IGNORE_TOKEN_ID = -100
RANK = int(os.environ.get("LOCAL_RANK") or 0)
random.seed(42)
from loguru import logger
from speedy import load_by_ext, dump_json_or_pickle


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    data_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # is_kilm: bool = field(default=False)
    use_lora: bool = field(default=False)
    target_loss_only: bool = field(default=False)
    batch_group_by_seq_length: bool = field(default=False)
    # avg_num_of_train_tokens: float = field(default=None)
    do_resume: bool = field(default=False)
    do_resize_token_embeddings: bool = field(default=False)

def collate_fn(
    batch, input_ids_pad_token_id, labels_ignore_token_id=-100, mask_pad_token_id=False
):
    assert len(batch) == 1
    inputs = batch[0]
    split_ids = inputs.pop("split_ids")
    new_inputs = {}
    for k in inputs:
        if isinstance(inputs[k], torch.Tensor):
            if k == "input_ids":
                pad_token_id = input_ids_pad_token_id
            elif k == "labels":
                pad_token_id = labels_ignore_token_id
            elif k == "attention_mask":
                pad_token_id = mask_pad_token_id
            else:
                raise NotImplementedError
            out = split_then_stack(inputs[k], split_ids, pad_token_id)
            new_inputs[k] = out

    new_inputs["loss_scale_factor"] = inputs["loss_scale_factor"]
    assert ( (new_inputs['labels']>0).sum(1) > 0).all() 
    return new_inputs


class DynamicBatchingTrainer(transformers.Trainer):

    def get_train_dataloader(self) -> DataLoader:
        return self.accelerator.prepare(
            DataLoader(
                self.train_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                collate_fn=lambda x: x[0],
            )
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        loss_scale_factor = inputs.pop("loss_scale_factor", 1)
        if not self.args.target_loss_only:
            inputs["labels"] = inputs["input_ids"].clone()
            pad_id = self.tokenizer.pad_token_id
            ids = inputs["labels"]==pad_id
            inputs["labels"][ids] = -100
            
        unscaled_loss = super().compute_loss(model, inputs, return_outputs)
        loss = unscaled_loss * loss_scale_factor
        self.prev_inputs = inputs
        if self.state.global_step % 10 == 0:
            input_num_tokens = inputs["attention_mask"].sum().item()
            target_num_tokens = inputs["labels"].gt(0).sum().item()
            pretty_log = f'[RANK={RANK}] Step: {self.state.global_step}, Target Loss Only: {self.args.target_loss_only}, Input Tokens: {input_num_tokens}, Target Tokens: {target_num_tokens}, Loss: {loss:.4f}, Loss Scale Factor: {loss_scale_factor:.4f}'
            logger.info(pretty_log)

        return loss


def split_then_stack(merged_ids, split_idxs, pad_value):
    # List to hold the split tensors
    batch_tensors = []

    # Previous split index, starting from 0
    prev_idx = 0
    for idx in split_idxs:
        # Split the tensor according to the split index
        batch = merged_ids[prev_idx:idx]
        batch_tensors.append(batch)
        prev_idx = idx

    # Find the longest length in the split tensors
    max_length = max(batch.size(0) for batch in batch_tensors)
    dtype = batch_tensors[0].dtype
    # Pad each tensor to have the same length as the longest tensor
    padded_batches = [
        torch.nn.functional.pad(
            batch, (0, max_length - batch.size(0)), "constant", pad_value
        )
        for batch in batch_tensors
    ]

    # Stack the padded tensors along a new dimension
    stacked_tensor = torch.stack(padded_batches).to(dtype=dtype)
    return stacked_tensor

from fast_hf_llm_trainer import create_chunks_with_train_tokens
class DynamicbatchingDataset(Dataset):
    def __init__(
        self, dataset, tokenizer_name:str, training_args:TrainingArguments,
    ):
        self.tokenizer_name = tokenizer_name
        self.data_max_length = training_args.data_max_length
        dataset.padding = False
        self.dataset = dataset
        self.pad_val = self.dataset.tokenizer.pad_token_id

        item_metas = self.compute_token_lengths(dataset)
        self.batches_with_split_points = create_chunks_with_train_tokens(item_metas, 
                                                                         self.data_max_length, 
            num_gpus=training_args.per_device_train_batch_size, accumulate_steps=training_args.gradient_accumulation_steps,
            target_loss_only=training_args.target_loss_only)


    def compute_token_lengths(self, ds):
        t = time.time()
        def _get_lens():
            lens = [raw_data[f"token_length_{self.tokenizer_name}"][0] for raw_data in ds.raw_data]
            num_train_tokens = [raw_data[f"token_length_{self.tokenizer_name}"][1] for raw_data in ds.raw_data]
            return lens, num_train_tokens

        lens, num_train_tokens = _get_lens()
        print(f"Finished getting lengths in {time.time()-t:.2f}s")
        item_metas = []
        for i, (l, n) in enumerate(zip(lens, num_train_tokens)):
            item_metas.append({"length": l, "num_train_token": n, "idx": i})
        return item_metas

    def __len__(self):
        return len(self.batches_with_split_points)

    def __merge_dict(self, list_d):
        d = list_d[0]
        todo_keys = [k for k in d if isinstance(d[k], torch.Tensor)]
        ret_d = {}
        for k in todo_keys:
            ret_d[k] = []

        for d in list_d:
            for k in todo_keys:
                ret_d[k].append(d[k])
        for k in todo_keys:
            ret_d[k] = torch.cat(ret_d[k])
        return ret_d

    def __getitem__(self, idx, counter=0):
        if counter > 10:
            raise Exception("Try to get item too many times (10 times)")
        batch_split_point = self.batches_with_split_points[idx]
        ids, split_ids, loss_scale_factor = [batch_split_point[k] for k in ["item_ids", "split_points", "loss_scale_factor"]]
        items = [self.dataset[idx] for idx in ids]
        item = self.__merge_dict(items)
        item["split_ids"] = split_ids
        item["loss_scale_factor"] = loss_scale_factor
        batch = collate_fn([item], self.pad_val, IGNORE_TOKEN_ID, False)
        # im_start_idx = self.dataset.tokenizer('<|im_start|>')['input_ids'][0]
        # assert batch['input_ids'][:,0].eq(im_start_idx).all(), f"invalide start token {batch['input_ids'][:,0]} != {im_start_idx}"
        return batch
