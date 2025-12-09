from unsloth import FastVisionModel
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import os

os.environ["UNSLOTH_VLLM_STANDBY"] = "1" # [NEW] Extra 30% context lengths!

# max_seq_length = 16384 # Must be this long for VLMs
max_seq_length = 32768 # Must be this long for VLMs
lora_rank = 16 # Larger rank = smarter, but slowe

model, tokenizer = FastVisionModel.from_pretrained(
    # model_name = "unsloth/Qwen2.5-VL-7B-Instruct",
    model_name = "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
    # model_name = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    gpu_memory_utilization = 0.8, # Reduce if out of memory
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True,  # False if not finetuning language layers
    finetune_attention_modules = True,  # False if not finetuning attention layers
    finetune_mlp_modules       = True,  # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)


# If not handled properly, prompt truncation may truncate image token
dataset = load_dataset("trl-internal-testing/zen-multi-image", "conversational_prompt_only", split="train")

dataset = dataset.filter(lambda x: len(x["images"]) > 0) # currently, mixing samples with and without images is not supported

def my_reward_function(prompts, completions, **kwargs):
    return [1.0] * len(prompts)

training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    log_completions = False,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 2, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = 1024,
    max_completion_length = 1024,
    num_train_epochs = 0.5, # Set to 1 for a full training run
    # max_steps = 60,
    save_steps = 60,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",

    # Below enables GSPO:
    importance_sampling_level = "sequence",
    mask_truncated_completions = False,
    loss_type = "dr_grpo",
)


trainer = GRPOTrainer(
    model = model,
    args = training_args,
    # Pass the processor to handle multimodal inputs
    processing_class = tokenizer,
    reward_funcs = [
        my_reward_function,
    ],
    train_dataset = dataset,
)

trainer.train()
