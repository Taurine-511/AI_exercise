from datasets import load_dataset, Dataset

user_prompt = """
You are given two images: a reference image (Image A) and a modified image (Image B).
Identify all visual differences between Image A and Image B.
Output each detected target's bbox coordinates in JSON format. The format of the bbox coordinates is:

[
    {"bbox_2d": [x1, y1, x2, y2], ”label”: ”target name”},
    {"bbox_2d": [x1, y1, x2, y2], ”label”: ”target name”},
    ...
]

If there are no such targets in the image, simply respond with empty list.
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
""".strip()


def prepare_dataset():
    dataset = load_dataset("Taurine511/spot_diff", split="train")

    messages = [
        {
            "role": "user",
            "content": [
                {"text": None, "type": "image"},
                {"text": None, "type": "image"},
                {"text": user_prompt, "type": "text"},
            ],
        }
    ]

    left_labels = []
    for labels in dataset["left_labels"]:
        result = []
        for label in labels:
            cx, cy, w, h = label["bbox"]
            bbox_2d = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
            bbox_2d = [x * 1000 for x in bbox_2d]
            result += [{"bbox_2d": bbox_2d}]
        left_labels.append(result)

    right_labels = []
    for labels in dataset["right_labels"]:
        result = []
        for label in labels:
            cx, cy, w, h = label["bbox"]
            bbox_2d = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
            bbox_2d = [x * 1000 for x in bbox_2d]
            result += [{"bbox_2d": bbox_2d}]
        right_labels.append(result)

    return Dataset.from_dict(
        {
            "prompt": [messages] * len(dataset),
            "images": dataset["images"],
            "left_labels": left_labels,
            "right_labels": right_labels,
        }
    )


def prepare_repeated_dataset():
    dataset = prepare_dataset()
    return Dataset.from_list([dataset[0]] * len(dataset))
