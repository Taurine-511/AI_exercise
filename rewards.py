import re


def format_reward(prompts, completions, **kwargs):
    scores = []

    format_pattern = re.compile(
        r"^\s*"
        r"<think>\s*(.+?)\s*</think>\s*"  # group 1: think block
        r"<answer>\s*(.+?)\s*</answer>\s*"  # group 2: answer block
        r"$",
        re.DOTALL,
    )

    for completion in completions:
        response = completion[0]["content"]
        match = format_pattern.match(response)

        if match and match.group(1).strip() and match.group(2).strip():
            score = 1.0
        else:
            score = -1.0

        scores.append(score)

    return scores
