from torch.utils.data import Dataset
from typing import Dict, List, Any
import torch
import transformers
from pydantic import BaseModel, validator

IGNORE_TOKEN_ID = -100
TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ 'system\nYou are a helpful assistant.\n' }}{% endif %}{{'' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ ''}}{% else %}{{ '\n' }}{% endif %}{% endfor %}"

class PreprocOutputModel(BaseModel):
    input_ids: List[int]
    target_ids: List[int]
    attention_mask: List[int]

    @validator('*', pre=True)
    def check_list_of_numbers(cls, v):
        if not all(isinstance(i, int) for i in v):
            raise ValueError('All elements must be numbers')
        return v

def default_preproc(
    messages: List[Dict[str, Any]],
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
) -> PreprocOutputModel:
    """Preprocesses the data for supervised fine-tuning."""
    text = tokenizer.apply_chat_template(
        messages,
        chat_template=TEMPLATE,
        tokenize=False,
        add_generation_prompt=False,
    )
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = tokens['input_ids'].squeeze(0).tolist()
    attention_mask = tokens['attention_mask'].squeeze(0).tolist()

    target_ids = [IGNORE_TOKEN_ID if token == tokenizer.pad_token_id else token for token in input_ids]

    return PreprocOutputModel(
        input_ids=input_ids,
        target_ids=target_ids,
        attention_mask=attention_mask
    )
from llm_utils import transform_messages_to_chatml
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        preprocess_fn=default_preproc,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.preproc_fn = preprocess_fn
        self.raw_data = raw_data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        ret: PreprocOutputModel = self.preproc_fn(
            transform_messages_to_chatml(self.raw_data[i]), self.tokenizer, self.max_len
        )
        return {
            "input_ids": torch.tensor(ret.input_ids, dtype=torch.int),
            "labels": torch.tensor(ret.target_ids, dtype=torch.long),
            "attention_mask": torch.tensor(ret.attention_mask, dtype=torch.int),
        }
