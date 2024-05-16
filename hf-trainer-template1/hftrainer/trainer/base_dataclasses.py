from dataclasses import dataclass, field
from typing import Optional, List
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = "Qwen1.5/Qwen-1_8B"


@dataclass
class DataArguments:
    data_path: str = None
    eval_data_path: str = None
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = None
    optim: str = "adamw_torch"
    model_max_length: int = 8192
    data_max_length: int = 8192
    use_lora: bool = False
    target_loss_only: Optional[str] = False
    learning_rate: float = 1e-5


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
