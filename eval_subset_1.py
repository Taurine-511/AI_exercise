import re
import ast
from openai import OpenAI
import json
import numpy as np
from tqdm.auto import tqdm 
import matplotlib.pyplot as plt
from datasets import load_dataset
from IPython.display import FileLink

output_file = "clevr_subset1_result_iou_num_dup.json"
output_img = "clevr_subset1_iou_num_dup.png"
INFER_JSON_PATH = "clever_subset1_iou_num_dup.json"



EVAL_INSTRUCTIONS_COUNT = """
You are an evaluator. Your ONLY task is to output a single integer.

Evaluation Rule:
1. Count how many unique facts from the [Reference] list are present in the [Candidate] list.
2. The score MUST NOT exceed the total number of items in the [Reference] list.
3. If multiple sentences in the [Candidate] describe the same single fact in the [Reference], count it as only 1 match.
4. Each item in the [Reference] can be marked as "covered" only once.
5. Be generous with paraphrasing, but strict about the total count.

Constraint (Strict):
1. The VERY FIRST character of your response must be a digit (0-9).
2. Do not include any preamble, thoughts, explanations, or "The number is:".
3. Do not use any markdown, code blocks, or punctuation.

[Reference]
{reference}

[Candidate]
{candidate}

Final Instruction: Output the integer now.
""".strip()

# ここだけ変更
BASE_URL = "URL"
API_KEY = "YOUR API KEY"


JUDGE_MODEL = "gpt-5-nano"
MODE = "count"
MAX_COMPLETION_TOKENS = 4096

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def format_list(items):
    return "\n".join(f"- {x}" for x in items)

def llm_judge(model_output: list[str], correct_answers: list[str], mode: str = "count", model: str = "gpt-5-nano"):
    assert mode in {"count"}

    user_prompt = f"""
    Reference (required facts):
    {format_list(correct_answers)}

    Candidate (model output):
    {format_list(model_output)}

    Please follow the evaluation task described above for the selected mode.
    """.strip()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EVAL_INSTRUCTIONS_COUNT},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )

    raw = (response.choices[0].message.content or "").strip()
    match = re.search(r'\d+', raw)
    if match:
        v = int(match.group())
        return v if v >= 0 else 0
    return 0

def extract_candidate_list(text: str):
    try:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end <= start:
            return []
        return ast.literal_eval(text[start:end])
    except Exception:
        return []
    
def extract_answer_text(text: str) -> str:
    if text is None:
        return ""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else text

with open(INFER_JSON_PATH, "r", encoding="utf-8") as f:
    infer = json.load(f)

per_problem = []
by_count = {}
total_correct = 0
total_wrong = 0
total_ref = 0
total_samples = 0

infer = infer['generations']

for rec in tqdm(infer, desc="Evaluating", unit="sample"):
    idx = int(rec["sample_id"])
    prompt_id = int(rec.get("prompt_id", -1))

    # ground truth
    correct = rec["label"]
    ref_len = len(correct)

    answer_str = extract_answer_text(rec.get("text", ""))
    generated_texts = [answer_str]
    n_samples = len(generated_texts)

    current_count = ref_len

    correct_sum = 0
    wrong_sum = 0

    for text in generated_texts:
        candidate_list = extract_candidate_list(text)
        score = llm_judge(candidate_list, correct, mode=MODE, model=JUDGE_MODEL)
        if score > ref_len:
            score = ref_len
        correct_sum += int(score)
        wrong_sum += int(ref_len - score)

    per_problem.append({
        "prompt_id": int(prompt_id),
        "count": int(current_count),
        "n_samples": int(n_samples),
        "reference_len": int(ref_len),
        "correct_sum": int(correct_sum),
        "wrong_sum": int(wrong_sum),
    })


    if current_count not in by_count:
        by_count[current_count] = {
            "correct_sum": 0,
            "wrong_sum": 0,
            "reference_len_sum": 0,
            "n_samples_sum": 0,
            "n_problems": 0,
        }
    by_count[current_count]["correct_sum"] += int(correct_sum)
    by_count[current_count]["wrong_sum"] += int(wrong_sum)
    by_count[current_count]["reference_len_sum"] += int(ref_len) * int(n_samples)
    by_count[current_count]["n_samples_sum"] += int(n_samples)
    by_count[current_count]["n_problems"] += 1

    total_correct += int(correct_sum)
    total_wrong += int(wrong_sum)
    total_ref += int(ref_len) * int(n_samples)
    total_samples += int(n_samples)

out = {
    "mode": MODE,
    "judge_model": JUDGE_MODEL,
    "summary": {
        "total_correct": int(total_correct),
        "total_wrong": int(total_wrong),
        "total_reference": int(total_ref),
        "total_samples": int(total_samples),
        "accuracy_like": float(total_correct / total_ref) if total_ref > 0 else 0.0,
    },
    "by_count": {str(k): v for k, v in sorted(by_count.items(), key=lambda x: x[0])},
    "per_problem": per_problem,
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=4)

x_axis = sorted(by_count.keys())
y_axis = []
for k in x_axis:
    denom = by_count[k]["reference_len_sum"]
    y_axis.append(by_count[k]["correct_sum"] / denom if denom > 0 else 0.0)

plt.figure(figsize=(8, 6))
plt.plot(x_axis, y_axis, marker="o", linestyle="-")
plt.xlabel("Count")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Count (All Samples)")
plt.grid(True)
plt.savefig(output_img)

print("Finished processing 350 samples.")
