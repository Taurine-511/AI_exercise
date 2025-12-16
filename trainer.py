import pathlib
from argparse import ArgumentParser

from unsloth import FastVisionModel  # isort: skip
from trl import GRPOConfig, GRPOTrainer  # type: ignore

import wandb
from config import Config
from data import prepare_dataset, prepare_repeated_dataset
from rewards import (
    format_reward,
    iou_reward,
    ioa_reward,
    unused_predict_penalty,
    duplicate_predict_penalty,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=pathlib.Path,
        help="configのパス",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    if cfg.use_wandb:
        wandb.init(
            entity="taurine511",
            project=cfg.project,
            name=cfg.run_name,
            config=vars(cfg),
        )

    # Model
    model, tokenizer = FastVisionModel.from_pretrained(**cfg.get_model_config())

    # LoRA
    model = FastVisionModel.get_peft_model(model, **cfg.get_lora_config())

    # Dataset
    dataset = prepare_dataset()

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        args=GRPOConfig(**cfg.get_training_config()),
        processing_class=tokenizer,
        reward_funcs=[iou_reward, unused_predict_penalty],
        train_dataset=dataset,
    )

    trainer.train()
