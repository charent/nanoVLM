import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import swanlab
import torch
import transformers
from accelerate.utils import DistributedType
from datasets import Features, Sequence, Value, load_dataset
from datasets.dataset_dict import DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerControl, TrainerState
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
)
from transformers.trainer_pt_utils import LabelSmoother

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

from data.collators import VQACollator
from data.datasets import VQADataset
from models.vision_language_model import VisionLanguageModel
from models.processors import get_tokenizer
from models.config import VLMConfig

torch.set_default_dtype(torch.bfloat16)


swanlab.init(
    logdir="./logs",
    mode="local",
)

# try:
#     import debugpy

#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as _:
#     pass


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    enable_flash_atten: Optional[bool] = field(default=False)
    enable_thinking: Optional[bool] = field(default=False)  # for Qwen3 etc


@dataclass
class DataArguments:
    train_file_dir: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_file_dir: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )


@dataclass
class TrainArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
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
    lora_modules_to_save: list[str] = field(
        default_factory=lambda: [
            "embed_tokens",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def get_arguments() -> (
    tuple[ModelArguments, DataArguments, TrainArguments, LoraArguments]
):

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainArguments, LoraArguments)
    )

    return parser.parse_args_into_dataclasses()


def train() -> None:
    global local_rank

    model_args, data_args, training_args, lora_args = get_arguments()

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    config = VLMConfig()

    tokenizer = get_tokenizer(
        config.lm_pretrain_path, config.vlm_extra_tokens, config.lm_chat_template
    )

    model: VisionLanguageModel = VisionLanguageModel.from_pretrained(
        model_args.model_name_or_path, config=config
    )

    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            inference_mode=False,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            # modules_to_save=lora_args.lora_modules_to_save,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.vision_encoder.enable_input_require_grads()
            model.decoder.enable_input_require_grads()

    # stage 1: training mp
    for param in model.vision_encoder.parameters():
        param.requires_grad = False

    # for param in model.decoder.parameters():
    #     param.requires_grad = False

    # Load data
    train_dataset = VQADataset(
        data_args.train_file_dir,
        config,
        train_max_token_length=training_args.model_max_length,
    )
    eval_dataset = None

    # init data_collator
    data_collator = VQACollator(tokenizer, max_length=training_args.model_max_length)

    training_args.lr_scheduler_kwargs = {"num_cycles": 0.5}
    training_args.remove_unused_columns = True
    training_args.dataloader_num_workers = 2

    # Start trainer
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    train()
