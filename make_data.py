import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import torch
from vllm import LLM, SamplingParams
from PIL import Image


# =========================
# dataset 想定
# dataset[i] = {
#   "prompt": ChatML形式の list[dict],
#   "images": [PIL.Image, PIL.Image]
# }
# =========================


def build_vllm_inputs(dataset, tokenizer) -> List[Dict]:
    inputs = []
    for sample in dataset:
        inputs.append(
            {
                "prompt": tokenizer.apply_chat_template(
                    sample["prompt"], add_generation_prompt=True, tokenize=False
                ),  # ChatML
                "multi_modal_data": {
                    "image": sample["images"],  # 画像2枚
                },
            }
        )
    return inputs


def main(dataset):
    # =========================
    # vLLM 初期化
    # =========================
    llm = LLM(
        model="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",  # 例: Qwen/Qwen2-VL-7B-Instruct
        trust_remote_code=True,
        quantization="awq",
        dtype="half",
        gpu_memory_utilization=0.88,
        tensor_parallel_size=torch.cuda.device_count(),
        limit_mm_per_prompt={"image": 2, "video": 0},
        max_model_len=32768,
    )

    # =========================
    # Sampling 設定
    # =========================
    sampling_params = SamplingParams(
        n=1,  # クエリあたりの生成数
        temperature=0.7,
        top_p=0.9,
        max_tokens=64,
    )

    # =========================
    # 入力構築
    # =========================
    tokenizer = llm.get_tokenizer()
    vllm_inputs = build_vllm_inputs(dataset, tokenizer)

    # =========================
    # 推論
    # =========================
    outputs = llm.generate(
        prompts=vllm_inputs,
        sampling_params=sampling_params,
    )

    # =========================
    # 結果をフラットな JSON 構造に変換
    # =========================
    results = []
    for prompt_id, out in enumerate(outputs):
        label = dataset[prompt_id].get("labels", None)
        for sample_id, cand in enumerate(out.outputs):
            results.append(
                {
                    "prompt_id": prompt_id,
                    "sample_id": sample_id,
                    "label": label,
                    "text": cand.text,
                }
            )

    # =========================
    # JSON 保存
    # =========================
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"vllm_generations_{timestamp}.json"

    payload = {
        "meta": {
            "model": llm.model_config.model,
            "num_prompts": len(dataset),
            "num_samples_per_prompt": sampling_params.n,
            "sampling_params": {
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "max_tokens": sampling_params.max_tokens,
            },
            "created_at": timestamp,
        },
        "generations": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} generations to {output_path}")


# =========================
# 実行例
# =========================
if __name__ == "__main__":
    from data import prepare_clever_dataset, prepare_dataset

    dataset = prepare_dataset()
    main(dataset)
