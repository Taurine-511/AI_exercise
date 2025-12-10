import pathlib
from dataclasses import asdict, dataclass

import yaml


@dataclass
class Config:
    # Model
    model_name: str = "unsloth/Qwen2.5-VL-7B-Instruct"
    max_seq_length: int = 32768
    load_in_4bit: bool = True
    fast_inference: bool = True
    gpu_memory_utilization: float = 0.8

    # LoRA
    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    random_state: int = 3407
    use_gradient_checkpointing: str = "unsloth"

    # Training
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    logging_steps: int = 1
    log_completions: bool = True
    gradient_accumulation_steps: int = 1
    num_generations: int = 8
    max_prompt_length: int = 1024
    max_completion_length: int = 1024
    num_train_epochs: float = 4.0
    max_grad_norm: float = 0.1
    report_to: str = "none"
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: pathlib.Path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        cfg = cls(**data)

        base_name = path.stem
        cfg.output_dir = f"outputs/{base_name}"
        cfg.learning_rate = float(cfg.learning_rate)

        return cfg

    def get_model_config(self):
        keys = [
            "model_name",
            "max_seq_length",
            "load_in_4bit",
            "fast_inference",
            "gpu_memory_utilization",
        ]
        full = asdict(self)
        print(full)
        return {k: full[k] for k in keys}

    def get_lora_config(self):
        keys = [
            "finetune_vision_layers",
            "finetune_language_layers",
            "finetune_attention_modules",
            "finetune_mlp_modules",
            "r",
            "lora_alpha",
            "lora_dropout",
            "bias",
            "random_state",
            "use_gradient_checkpointing",
        ]
        full = asdict(self)
        return {k: full[k] for k in keys}

    def get_training_config(self):
        keys = [
            "learning_rate",
            "adam_beta1",
            "adam_beta2",
            "weight_decay",
            "warmup_ratio",
            "lr_scheduler_type",
            "optim",
            "logging_steps",
            "log_completions",
            "gradient_accumulation_steps",
            "num_generations",
            "max_prompt_length",
            "max_completion_length",
            "num_train_epochs",
            "max_grad_norm",
            "report_to",
            "output_dir",
        ]
        full = asdict(self)
        return {k: full[k] for k in keys}
