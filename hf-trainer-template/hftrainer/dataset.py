# src/dataset.py
from torch.utils.data import Dataset
import torch
from transformers import PreTrainedTokenizer

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer: PreTrainedTokenizer, max_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        inputs = self.tokenizer.encode_plus(
            item['text'],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }
