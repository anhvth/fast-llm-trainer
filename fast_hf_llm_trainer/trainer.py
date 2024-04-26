import transformers
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math
import torch
import time
import os
import json
# from dataset_factory.make_dataset import LazySupervisedDataset
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

    new_inputs["scale_factor"] = inputs["scale_factor"]
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
        t = time.time()
        scale_factor = inputs.pop("scale_factor", 1)
        unscaled_loss = super().compute_loss(model, inputs, return_outputs)
        loss = unscaled_loss * scale_factor
        self.prev_inputs = inputs
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


class DynamicbatchingDataset(Dataset):
    def __init__(
        self, dataset, data_max_length, tokenizer_name, pad_val=-1, force_recreate_batches=False
    ):
        self.tokenizer_name = tokenizer_name
        self.data_max_length = data_max_length
        dataset.padding = False
        self.dataset = dataset
        self.force_recreate_batches = force_recreate_batches
        self.pad_val = pad_val
        self.item_metas = self.__get_random_batch_ids_with_splits(
            dataset, data_max_length
        )


    def get_lengths(self, ds):
        t = time.time()
        from speedy import identify, memoize

        def _get_lens():
            lens = [raw_data[f"token_length_{self.tokenizer_name}"][0] for raw_data in ds.raw_data]
            num_train_tokens = [raw_data[f"token_length_{self.tokenizer_name}"][1] for raw_data in ds.raw_data]
            return lens, num_train_tokens

        # name = identify(ds.raw_data)
        self.name = ds.dataset_id
        lens, num_train_tokens = _get_lens()
        print(f"Finished getting lengths in {time.time()-t:.2f}s")
        item_metas = []
        for i, (l, n) in enumerate(zip(lens, num_train_tokens)):
            item_metas.append({"length": l, "num_train_token": n, "idx": i})
        return item_metas

    def __get_random_batch_ids_with_splits(self, dataset, max_length):
        item_metas = self.get_lengths(self.dataset)
        idx_to_num_train_tokens = {
            item["idx"]: item["num_train_token"] for item in item_metas
        }

        tmp_file = f"/tmp/random_batch_ids{self.name}_{max_length}.pkl"
        from speedy import Clock
        clock = Clock()
        if (not osp.exists(tmp_file) and RANK == 0) or self.force_recreate_batches:
            len_to_indexes = {}
            for item_meta in item_metas:
                length = item_meta["length"]
                index = item_meta["idx"]
                if item_meta["num_train_token"] == 0:
                    continue
                if length <= max_length:
                    if length not in len_to_indexes:
                        len_to_indexes[length] = []
                    len_to_indexes[length].append(index)

            batches_with_split_points = []
            pbar = tqdm(total=len(dataset))
            sorted_lengths = np.array(sorted(len_to_indexes.keys()))
            while len(len_to_indexes) > 0:
                pbar.update(1)
                clock.print_table(every=1)
                current_batch_indexes = []
                current_split_points = []
                len_left = max_length
                accumulated_length = (
                    0  # To keep track of the accumulated lengths for split points
                )
                first_item_len = None
                max_item_len = 0
                clock.start()
                while len(len_to_indexes) > 0:
                    cond1 = sorted_lengths <= len_left
                    cond2 = np.abs(sorted_lengths - first_item_len) < 64 if first_item_len is not None else None
                    available_lengths = sorted_lengths[
                        np.logical_and(cond1, cond2) if cond2 is not None else cond1
                    ]
                    if len(available_lengths) == 0:
                        break
                    clock.update("get_available_lengths")
                    chosen_length = random.choice(available_lengths)
                    _id = np.random.choice(len(len_to_indexes[chosen_length]))
                    chosen_index = len_to_indexes[chosen_length].pop(_id)
                    clock.update("random_choice")
                    if first_item_len is None:
                        first_item_len = chosen_length
                        max_item_len = max(max_item_len, chosen_length)
                    clock.update("update_first_item_len")
                    current_batch_indexes.append(chosen_index)
                    accumulated_length += chosen_length
                    current_split_points.append(accumulated_length)
                    len_left = max_length - max_item_len * len(current_batch_indexes)*1.2
                    pbar.update(1)
                    clock.update("update_len_left")
                    if len(len_to_indexes[chosen_length]) == 0:
                        del len_to_indexes[
                            chosen_length
                        ]  
                        sorted_lengths = sorted_lengths[sorted_lengths != chosen_length]
                    clock.update("while_loop")
                if current_batch_indexes:
                    batches_with_split_points.append(
                        (current_batch_indexes, current_split_points)
                    )
                    first_item_len = None

            dump_json_or_pickle(batches_with_split_points, tmp_file)

        while not osp.exists(tmp_file):
            time.sleep(RANK + 1)
            print(f"[{RANK=}]Waiting for {tmp_file}")
        for i in range(10):
            try:
                batches_with_split_points = load_by_ext(tmp_file)
                print(f"[{RANK=}]Loaded {tmp_file}")
                break
            except:
                time.sleep(2 * (i + 1))
                print(f"[{RANK=}] Error loading {tmp_file}, retrying {i+1}/10")

        batches = []
        num_gpus = int(os.environ["GPUS_PER_NODE"])
        # filter out item with no train tokens
        batches_with_split_points = self.__group_by_seq_length(
            batches_with_split_points, num_gpus, shuffle=True, drop_last=False
        )
        global_bz = int(os.environ["GPUS_PER_NODE"]) * int(
            os.environ["ACCUMULATE_STEP"]
        )

        for i in range(0, len(batches_with_split_points), global_bz):
            global_batch_items = batches_with_split_points[i : i + global_bz]
            # Get number of train tokens for each item in the global batch
            _all_ids_flat = [
                item for sublist in global_batch_items for item in sublist[0]
            ]
            num_train_tokens_total = sum(
                [idx_to_num_train_tokens[idx] for idx in _all_ids_flat]
            )
            avg_train_token_in_this_global_batch = num_train_tokens_total / global_bz
            new_data = []
            for ids, split_points in global_batch_items:
                train_tokens = []
                for idx in ids:
                    train_tokens.append(idx_to_num_train_tokens[idx])
                train_tokens = sum(train_tokens)
                loss_scale_factor = train_tokens / avg_train_token_in_this_global_batch
                new_data.append([ids, split_points, loss_scale_factor])
            batches.append(new_data)

        batches_with_split_points = [item for sublist in batches for item in sublist]
        return batches_with_split_points

    def __group_by_seq_length(self, items, bz, shuffle=True, drop_last=True):
        to_be_removed = []
        for i, item in enumerate(items):
            if len(item[0]) == 0:
                to_be_removed.append(i)

        batches = []
        items = sorted(items, key=lambda x: len(x[0]))
        for i in range(0, len(items), bz):
            batch = items[i : i + bz]
            if len(batch) < bz and drop_last:
                continue
            batches.append(batch)
        if shuffle:
            random.shuffle(batches)
        # flatten
        batches = [item for sublist in batches for item in sublist]
        return batches

    def __len__(self):
        return len(self.item_metas)

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
        ids, split_ids, scale_factor = self.item_metas[idx]
        items = [self.dataset[idx] for idx in ids]
        item = self.__merge_dict(items)
        item["split_ids"] = split_ids
        item["scale_factor"] = scale_factor
        batch = collate_fn([item], self.pad_val, IGNORE_TOKEN_ID, False)
        # (dds[0]['input_ids'][:,0] == tokenizer('<|im_start|>', return_tensors='pt')['input_ids'][0,0]).all()
        im_start_idx = self.dataset.tokenizer('<|im_start|>')['input_ids'][0]
        assert batch['input_ids'][:,0].eq(im_start_idx).all(), f"invalide start token {batch['input_ids'][:,0]} != {im_start_idx}"
        return batch
        # _item = deepcopy(item)
        # except Exception as e:
        #     out_dir= '.cache/debug'
        #     outfile = f'{out_dir}/error_{idx}_RANK-{RANK}.pkl'
        #     _item['error'] = str(e)
        #     dump_json_or_pickle(_item, outfile)
        #     if RANK == 0 and counter==0:
        #         import traceback; traceback.print_exc()
        #     print(f"RANK: {RANK}, ERROR: {e}, idx: {idx}, outfile: {outfile}")
        #     return self.__getitem__(idx + 1, counter=counter+1)