import random
from speedy import Clock
import transformers
from llm_utils import transform_messages, load_chat_dataset
from copy import deepcopy
from transformers.trainer_pt_utils import LabelSmoother
from typing import *
import torch
import numpy as np
from loguru import logger

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
# TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{% endif %}{% if loop.last %}{% else %}{% endif %}{% endfor %}"
TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
def preprocess_chatlm_to_tokens(
    messages:List[dict[str,str]],
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int = None,
    target_with_assistant=True,
) -> Dict:
    """
    Formats input messages for training a language model.

    Args:
        messages (List[dict[str,str]]): List of dictionaries representing the input messages.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used to tokenize the messages.
        max_len (int, optional): Maximum length of the input messages. Defaults to None.
        target_loss_only (bool, optional): Whether to include only the target loss in the output. Defaults to False.

    Returns:
        Dict: A dictionary containing the formatted input messages, including input_ids, target_ids, attention_mask, labels, length, and num_train_tokens.

    Raises:
        AssertionError: If the tokenizer does not have the pad_token_id attribute.

    """
    assert messages[0]['role'] in ['system', 'user'], "First message must be system or user"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.chat_template is None:
        tokenizer.chat_template = TEMPLATE
    assert hasattr(tokenizer, 'pad_token_id'), "Tokenizer must have pad_token_id attribute"
    input_ids = []
    target_ids = []

    input_id = []
    target_with_assistant = []
    for i in range(len(messages)):
        _ids = tokenizer.apply_chat_template(
            [messages[i]], tokenize=True, add_special_tokens=False)
        input_id += _ids

        if messages[i]['role'] == 'assistant':
            target_with_assistant += _ids
        else:
            target_with_assistant += [IGNORE_TOKEN_ID]*len(_ids)


    # maxlen
    input_id = input_id[:max_len]
    target_with_assistant = target_with_assistant[:max_len]
    # to tensor
    input_ids = torch.tensor([input_id], dtype=torch.long)
    target_ids = torch.tensor([target_with_assistant], dtype=torch.long)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return dict(
        input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask, labels=target_ids,
        length=len(input_id),
        num_train_tokens=target_ids.ne(IGNORE_TOKEN_ID).sum().item(),
    )
def group_batches_by_sequence_length(items, bz, shuffle=True, drop_last=True):
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

def get_available_sequence_lengths(sorted_lengths, len_left, first_item_len):
    cond1 = sorted_lengths <= len_left
    cond2 = np.abs(sorted_lengths - first_item_len) < 64 if first_item_len is not None else None
    available_lengths = sorted_lengths[np.logical_and(cond1, cond2) if cond2 is not None else cond1]
    return available_lengths

def update_length_to_indexes_mapping(len_to_indexes, chosen_length, sorted_lengths):
    if len(len_to_indexes[chosen_length]) == 0:
        del len_to_indexes[chosen_length]
        sorted_lengths = sorted_lengths[sorted_lengths != chosen_length]
    return len_to_indexes, sorted_lengths

def create_batches_with_split_points(item_metas, max_length):
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
    sorted_lengths = np.array(sorted(len_to_indexes.keys()))
    while len(len_to_indexes) > 0:
        current_batch_indexes = []
        current_split_points = []
        len_left = max_length
        accumulated_length = 0
        first_item_len = None
        max_item_len = 0

        while len(len_to_indexes) > 0:
            available_lengths = get_available_sequence_lengths(sorted_lengths, len_left, first_item_len)
            if len(available_lengths) == 0:
                break
            chosen_length = random.choice(available_lengths)
            _id = np.random.choice(len(len_to_indexes[chosen_length]))
            chosen_index = len_to_indexes[chosen_length].pop(_id)
            if first_item_len is None:
                first_item_len = chosen_length
                max_item_len = max(max_item_len, chosen_length)
            current_batch_indexes.append(chosen_index)
            accumulated_length += chosen_length
            current_split_points.append(accumulated_length)
            len_left = max_length - max_item_len * len(current_batch_indexes) * 1.2
            len_to_indexes, sorted_lengths = update_length_to_indexes_mapping(len_to_indexes, chosen_length, sorted_lengths)

        if current_batch_indexes:
            batches_with_split_points.append((current_batch_indexes, current_split_points))
            first_item_len = None

    return batches_with_split_points

def create_chunks_with_train_tokens(item_metas, max_length, num_gpus, accumulate_steps, target_loss_only):
    """Create chunks with train tokens.

    This function creates chunks with train tokens based on the provided item metadata.

    Args:
        item_metas (List[Dict]): A list of dictionaries representing the metadata for each item. Each dictionary should contain the following keys:
            - 'idx' (int): The index of the item.
            - 'num_train_token' (int): The number of train tokens in the item.
            - 'length' (int): The length of the item (used when 'num_train_token' is not available).
        max_length (int): The maximum length of each chunk.
        num_gpus (int): The number of GPUs.
        accumulate_steps (int): The number of steps to accumulate gradients.
        target_loss_only (bool, optional): Whether to only consider items with 'num_train_token' when calculating the loss. Defaults to False.

    Returns:
        List[Dict]: A list of dictionaries representing the batches with split points. Each dictionary contains the following keys:
            - 'item_ids' (List[int]): The IDs of the items in the batch.
            - 'split_points' (List[int]): The split points for each item in the batch.
            - 'loss_scale_factor' (float): The loss scale factor for the batch.
    """
    if target_loss_only:
        idx_to_num_train_tokens = {item["idx"]: item["num_train_token"] for item in item_metas}
    else:
        idx_to_num_train_tokens = {item["idx"]: item["length"] for item in item_metas}

    batches_with_split_points = create_batches_with_split_points(item_metas, max_length)

    batches = []
    batches_with_split_points = group_batches_by_sequence_length(
        batches_with_split_points, num_gpus, shuffle=True, drop_last=False
    )
    global_bz = num_gpus * accumulate_steps
    for i in range(0, len(batches_with_split_points), global_bz):
        global_batch_items = batches_with_split_points[i : i + global_bz]
        _all_ids_flat = [item for sublist in global_batch_items for item in sublist[0]]
        num_train_tokens_total = sum([idx_to_num_train_tokens[idx] for idx in _all_ids_flat])
        avg_train_token_in_this_global_batch = num_train_tokens_total / global_bz
        new_data = []
        for ids, split_points in global_batch_items:
            train_tokens = sum([idx_to_num_train_tokens[idx] for idx in ids])
            loss_scale_factor = train_tokens / avg_train_token_in_this_global_batch
            new_data.append({'item_ids': ids, 'split_points': split_points, 'loss_scale_factor': loss_scale_factor})
        batches.append(new_data)
    batches_with_split_points = [item for sublist in batches for item in sublist]
    return batches_with_split_points