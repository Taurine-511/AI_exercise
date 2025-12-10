import pathlib
from argparse import ArgumentParser

from unsloth import FastVisionModel
from trl import GRPOTrainer, GRPOConfig # type: ignore # unslothはtrl.trainer.grpo_trainerとかをパッチしないため

from config import Config
from data import prepare_dataset
from rewards import format_reward

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
        reward_funcs=[format_reward],
        train_dataset=dataset,
    )

    trainer.train()
