import sys


from dataclasses import dataclass, field
import os
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from speedy import dump_json_or_pickle, logger
from llm_utils import load_chat_dataset
import pandas as pd
from speedy import multi_process

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# from dataset_factory.make_dataset import (
#     make_supervised_data_module,
# )
from fast_hf_llm_trainer import preprocess_chatlm_to_tokens
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--fold", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--no_overwrite_cache", action="store_true")
    parser.add_argument("--data_max_length", type=int, required=True)
    args = parser.parse_args()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if hasattr(tokenizer, "eos_token_id"):
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif hasattr(tokenizer, "eod_id"):
            tokenizer.pad_token_id = tokenizer.eod_id
        else:
            raise ValueError("Tokenizer does not have pad_token_id")
    assert args.tokenizer_name is not None
    from llm_utils import transform_messages_to_chatml
    def item_to_len(item):
        item_chatml = transform_messages_to_chatml(item)
        ret = preprocess_chatlm_to_tokens(
            item_chatml,
            tokenizer,
            max_len=args.data_max_length,
        )
        item[f'token_length_{args.tokenizer_name}'] = [ret['length'], ret['num_train_tokens']]
        return item
    if args.data_path.endswith('.tsv'):
        path_df = pd.read_csv(args.data_path, header=None, sep='\t', names=['path', 'epoch_percent'])
    # a single .json file
    elif args.data_path.endswith('.json') and os.path.exists(args.data_path):
        path_df = pd.DataFrame([{'path': args.data_path, 'epoch_percent': 100}])
    for i, row in path_df.iterrows():
        path = row['path']
        if path.startswith('#'):continue
        print(f'{i}/{len(path_df)}-', path)
        if not os.path.exists(path):
            print('Warning: path does not exist', path)
            continue
        items = load_chat_dataset(path, return_format='sharegpt')
        item_to_len(items[0])
        items = multi_process(item_to_len, items, os.cpu_count()-1)
        items = [item for item in items if item is not None]
        assert len(items) > 0, f'No valid data found {path}'
        dump_json_or_pickle(items, path)
        logger.info('Processed', path)

if __name__ == "__main__":
    main()
