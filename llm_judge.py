from openai import OpenAI

EVAL_INSTRUCTIONS_ALL = """
Task: Compare two unordered lists of sentences:
- Reference: required facts
- Candidate: model output

Decide whether every required fact in Reference is expressed somewhere in Candidate.
Paraphrases are allowed if the meaning is preserved.
Each reference fact can be counted at most once.
If a required fact is missing or contradicted, it is not covered.

Return only:
True  (if all required facts are covered)
False (otherwise)
""".strip()

EVAL_INSTRUCTIONS_COUNT = """
Task: Compare two unordered lists of sentences:
- Reference: required facts
- Candidate: model output

Decide whether every required fact in Reference is expressed somewhere in Candidate.
Paraphrases are allowed if the meaning is preserved.
Each reference fact can be counted at most once.
If a required fact is missing or contradicted, it is not covered.

Return only:
a single non-negative integer representing the count
""".strip()

BASE_URL = "ENTER_ENDPOINT"
API_KEY = "ENTER YOUR API_KEY"


def llm_judge(
    model_output: list[str],
    correct_answers: list[str],
    mode: str = "all",  # "all" or "count"
    model: str = "gpt-5-mini"
):
    assert mode in {"all", "count"}

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def format_list(items):
        return "\n".join(f"- {x}" for x in items)
    
    user_prompt = f"""
    [Correct Answers]
    {format_list(correct_answers)}

    [Model Output]
    {format_list(model_output)}

    Perform the task according to the mode.
    """.strip()

    if(mode=='all'):
        system_prompt = EVAL_INSTRUCTIONS_ALL
    else:
        system_prompt = EVAL_INSTRUCTIONS_COUNT
        
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=5,
    )

    raw = (response.choices[0].message.content or "").strip()

    # Strong output validation
    if mode == "all":
        return raw == "True"

    # mode == "count"
    try:
        return int(raw)
    except ValueError:
        return 0
    
# test
def main():
    # Correct answers: unordered reference set
    correct_answers = [
        'the big cyan metal sphere in front of the big matte cube changed to brown',
        'the big cyan metal ball that is in front of the large cyan rubber sphere turned brown',
        'the big cyan metal sphere in front of the purple shiny ball turned brown',
        'the large cyan metallic ball in front of the tiny purple object turned brown'
    ]

    # Model output: list of sentences from the VLM
    model_output = [
        "the tiny purple rubber ball to the right of the large gray ball is gone",
        "the tiny purple matte sphere to the right of the small blue cube has disappeared",
        "the tiny purple rubber sphere in front of the shiny sphere is gone",
    ]

    # Test ALL mode
    result_all = llm_judge(
        model_output=model_output,
        correct_answers=correct_answers,
        mode="all",
    )
    print("MODE = ALL")
    print("Expected: False")
    print("Actual  :", result_all)
    print("-" * 40)

    # Test COUNT mode
    result_count = llm_judge(
        model_output=model_output,
        correct_answers=correct_answers,
        mode="count",
    )
    print("MODE = COUNT")
    print("Expected: 1 (likely)")
    print("Actual  :", result_count)
    print("-" * 40)


if __name__ == "__main__":
    main()