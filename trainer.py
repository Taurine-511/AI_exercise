import pathlib
from argparse import ArgumentParser

from unsloth import FastVisionModel  # isort: skip
from dotenv import load_dotenv
from trl import GRPOConfig, GRPOTrainer  # type: ignore

load_dotenv()

import wandb
from config import Config
from data import prepare_dataset
from rewards import iou_reward, duplicate_predict_penalty, unused_predict_penalty


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
        reward_funcs=[iou_reward, duplicate_predict_penalty, unused_predict_penalty],
        train_dataset=dataset,
    )

    trainer.train()
