from hftrainer.datasets.lazy import LazySupervisedDataset
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
from fast_hf_llm_trainer import create_chunks_with_train_tokens
from hftrainer.base import BaseTrainer
from llm_utils import load_chat_dataset
from speedy import load_by_ext

from loguru import logger

from hftrainer.utils import rank0_log_info


# from modeling.dynamic_batching_trainer import split_then_stack
IGNORE_TOKEN_ID = -100
RANK = int(os.environ.get("LOCAL_RANK") or 0)
random.seed(42)


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
    target_loss_only = False


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
    assert ((new_inputs["labels"] > 0).sum(1) > 0).all()
    return new_inputs


class DynamicBatchingTrainer(BaseTrainer):

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
        if not self.training_args.target_loss_only:
            inputs["labels"] = inputs["input_ids"].clone()
            pad_id = self.tokenizer.pad_token_id
            ids = inputs["labels"] == pad_id
            inputs["labels"][ids] = -100

        unscaled_loss = super().compute_loss(model, inputs, return_outputs)
        loss = unscaled_loss * loss_scale_factor
        self.prev_inputs = inputs
        if self.state.global_step % 10 == 0:
            input_num_tokens = inputs["attention_mask"].sum().item()
            target_num_tokens = inputs["labels"].gt(0).sum().item()
            pretty_log = f"[RANK={RANK}] Step: {self.state.global_step}, Target Loss Only: {self.training_args.target_loss_only}, Input Tokens: {input_num_tokens}, Target Tokens: {target_num_tokens}, Loss: {loss.item():.4f}, Loss Scale Factor: {loss_scale_factor:.4f}"
            rank0_log_info(pretty_log)

        return loss

    def load_datasets(self):

        data = load_by_ext(self.data_args.data_path)
        ds = LazySupervisedDataset(
            data, self.tokenizer, self.training_args.data_max_length
        )
        dataset = DynamicbatchingDataset(ds, "qwen", self.training_args)
        return dataset, None  # train/val


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


class DynamicbatchingDataset(Dataset):
    """# DynamicbatchingDataset Class Documentation

    ## Overview
    The `DynamicbatchingDataset` class is a custom dataset designed for dynamic batching, optimized for training with variable-length sequences using the Hugging Face `transformers` library. It organizes data into batches based on token lengths and handles the creation of these batches to optimize GPU utilization during training.

    ## Initialization
    ### `__init__(self, dataset, tokenizer_name: str, training_args: TrainingArguments)`

    #### Parameters:
    - `dataset`: The dataset to be batched, typically an instance of a Hugging Face Dataset object.
    - `tokenizer_name` (str): The name of the tokenizer used to tokenize the dataset.
    - `training_args` (TrainingArguments): The training arguments containing parameters such as maximum data length, batch size, and gradient accumulation steps.

    #### Description:
    Initializes the `DynamicbatchingDataset` instance, setting up necessary parameters, computing token lengths for each item in the dataset, and creating batches based on these token lengths.

    ## Methods

    ### `compute_token_lengths(self, ds)`

    #### Parameters:
    - `ds`: The dataset whose token lengths need to be computed.

    #### Returns:
    - `item_metas` (List[Dict]): A list of dictionaries, each containing the length, number of training tokens, and index of each item in the dataset.

    #### Description:
    Computes the token lengths and the number of training tokens for each item in the dataset. This information is used to create batches that maximize GPU efficiency.

    ### `__len__(self)`

    #### Returns:
    - `length` (int): The number of batches created.

    #### Description:
    Returns the number of batches created based on the token lengths and batching strategy.

    ### `__merge_dict(self, list_d)`

    #### Parameters:
    - `list_d` (List[Dict]): A list of dictionaries where each dictionary represents an item in the dataset.

    #### Returns:
    - `ret_d` (Dict): A single dictionary with concatenated tensor values from the input list.

    #### Description:
    Merges a list of dictionaries into a single dictionary by concatenating the tensor values. This is used to prepare the items for batching.

    ### `__getitem__(self, idx, counter=0)`

    #### Parameters:
    - `idx`: The index of the batch to retrieve.
    - `counter` (int, optional): A counter to prevent infinite retries (default is 0).

    #### Returns:
    - `batch` (Dict): A batch of data prepared for training.

    #### Description:
    Retrieves a batch of data based on the provided index. The method handles fetching the item IDs and split points for the batch, merging the items, and preparing the final batch using a custom collate function. If the method fails more than 10 times, it raises an exception to avoid infinite retries.

    ### Example Usage

    ```python
    from transformers import TrainingArguments

    # Assume `dataset` is a pre-loaded Hugging Face dataset and `tokenizer` is a pre-initialized tokenizer.
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        data_max_length=512,
        target_loss_only=True
    )

    dynamic_dataset = DynamicbatchingDataset(
        dataset=dataset,
        tokenizer_name=tokenizer.name_or_path,
        training_args=training_args
    )

    # Accessing a batch
    batch = dynamic_dataset[0]
    ```

    ### Internal Methods
    - `__merge_dict(self, list_d)`: Merges a list of dictionaries into one by concatenating their tensor values.
    - `__getitem__(self, idx, counter=0)`: Retrieves and processes a batch of data for the given index.

    ## Notes
    - The `compute_token_lengths` method is crucial for determining how the dataset is split into batches.
    - The class heavily relies on the `TrainingArguments` for configuring batch sizes, accumulation steps, and other training parameters.
    - The custom collate function, `collate_fn`, is used to handle padding and other preprocessing steps required for batching.

    By using this class, users can efficiently handle dynamic batching of variable-length sequences, improving the efficiency of training models on datasets with diverse sequence lengths.
    """

    def __init__(
        self,
        dataset,
        tokenizer_name: str,
        training_args: TrainingArguments,
    ):
        self.tokenizer_name = tokenizer_name
        self.data_max_length = training_args.data_max_length
        self.dataset = dataset
        self.pad_val = self.dataset.tokenizer.pad_token_id

        item_metas = self.compute_token_lengths(dataset)
        self.batches_with_split_points = create_chunks_with_train_tokens(
            item_metas,
            self.data_max_length,
            num_gpus=training_args.per_device_train_batch_size,
            accumulate_steps=training_args.gradient_accumulation_steps,
            target_loss_only=training_args.target_loss_only,
        )

    def compute_token_lengths(self, ds):
        t = time.time()

        def _get_lens():
            lens = [
                raw_data[f"token_length_{self.tokenizer_name}"][0]
                for raw_data in ds.raw_data
            ]
            num_train_tokens = [
                raw_data[f"token_length_{self.tokenizer_name}"][1]
                for raw_data in ds.raw_data
            ]
            return lens, num_train_tokens

        lens, num_train_tokens = _get_lens()
        rank0_log_info(f"Finished getting lengths in {time.time()-t:.2f}s")
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
        ids, split_ids, loss_scale_factor = [
            batch_split_point[k]
            for k in ["item_ids", "split_points", "loss_scale_factor"]
        ]
        items = [self.dataset[idx] for idx in ids]
        item = self.__merge_dict(items)
        item["split_ids"] = split_ids
        item["loss_scale_factor"] = loss_scale_factor
        try:
            batch = collate_fn([item], self.pad_val, IGNORE_TOKEN_ID, False)
            return batch
        except RuntimeError as e:
            return self.__getitem__(idx, counter + 1)