import re
import json

from utils import score_max_iou, score_max_ioa


def format_reward(prompts, completions, **kwargs):
    scores = []

    format_pattern = re.compile(
        r"^\s*"
        r"<think>\s*(.+?)\s*</think>\s*"      # group 1: think block
        r"<answer>\s*(.+?)\s*</answer>\s*"    # group 2: answer block
        r"$",
        re.DOTALL,
    )

    for completion in completions:
        response = completion[0]["content"]
        match = format_pattern.match(response)

        if not match:
            scores.append(0.0)
            continue

        think_block = match.group(1).strip()
        answer_block = match.group(2).strip()

        # think と answer の両方が空ならNG
        if not think_block or not answer_block:
            scores.append(0.0)
            continue

        # answer を JSON としてロード
        try:
            json.loads(answer_block)
            scores.append(1.0)
        except json.JSONDecodeError:
            scores.append(0.0)

    return scores


def iou_reward(prompts, completions, **kwargs):
    right_labels = kwargs["right_labels"]
    scores = []

    format_pattern = re.compile(
        r"^\s*"
        r"<think>\s*(.+?)\s*</think>\s*"      # group 1: think block
        r"<answer>\s*(.+?)\s*</answer>\s*"    # group 2: answer block
        r"$",
        re.DOTALL,
    )

    for completion, labels in zip(completions, right_labels):
        response = completion[0]["content"]
        match = format_pattern.match(response)

        if not match:
            scores.append(0.0)
            continue

        answer_block = match.group(2).strip()

        # answer を JSON としてロード
        try:
            answer = json.loads(answer_block)
            score = score_max_iou(answer, labels) / len(labels)
            scores.append(score)
        except json.JSONDecodeError:
            scores.append(0.0)

    return scores


def ioa_reward(prompts, completions, right_labels, **kwargs):
    scores = []

    format_pattern = re.compile(
        r"^\s*"
        r"<think>\s*(.+?)\s*</think>\s*"      # group 1: think block
        r"<answer>\s*(.+?)\s*</answer>\s*"    # group 2: answer block
        r"$",
        re.DOTALL,
    )

    for completion, labels in zip(completions, right_labels):
        response = completion[0]["content"]
        match = format_pattern.match(response)

        if not match:
            scores.append(0.0)
            continue

        answer_block = match.group(2).strip()

        # answer を JSON としてロード
        try:
            answer = json.loads(answer_block)
            score = score_max_ioa(answer, labels) / len(labels)
            scores.append(score)
        except json.JSONDecodeError:
            scores.append(0.0)

    return scores

