from speedy import setup_logger, logger

setup_logger("./logs")
import os
from speedy import *

os.environ["JUPYTER"] = "True"
from speedy import *
from fast_hf_llm_trainer import DynamicBatchingTrainer
if __name__ == "__main__":
    trainer = DynamicBatchingTrainer("config/template_args.yaml")
    trainer.train()
