from speedy import load_by_ext, dump_json_or_pickle, setup_logger, logger

setup_logger('./logs')
import os
from speedy import *
from hftrainer.trainer.datasets import LazySupervisedDataset
os.environ["JUPYTER"] = "True"
from hftrainer.trainer.base import *
from speedy import *
from fast_hf_llm_trainer import DynamicBatchingTrainer
trainer = DynamicBatchingTrainer('hf-trainer-template/config/template_args.yaml')
trainer.train()