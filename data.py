from datasets import Dataset, load_dataset
from qwen_utils import convert_to_qwen25vl_format

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

clever_template = """
{instruction}

Output each detected differences in JSON format. The format of the output is:

[
    "the small metal sphere turned rubber",
    "the small red metal sphere to the left of the purple cylinder became matte",
    ...
]

If there are no such targets in the image, simply respond with empty list.
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
""".strip()


def prepare_dataset():
    def format_example(example):
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
        image_left, image_right = example["images"]
        width_left, height_left = image_left.size
        width_right, height_right = image_right.size

        left_labels = []
        for label in example["left_labels"]:
            cx, cy, w, h = label["bbox"]
            bbox_2d = [
                (cx - w / 2) * width_left,
                (cy - h / 2) * height_left,
                (cx + w / 2) * width_left,
                (cy + h / 2) * height_left,
            ]
            left_labels += [
                {
                    "bbox_2d": convert_to_qwen25vl_format( # qwen2.5用にbboxをリサイズする
                        bbox_2d, height_left, width_left
                    )
                }
            ]

        right_labels = []
        for label in example["right_labels"]:
            cx, cy, w, h = label["bbox"]
            bbox_2d = [
                (cx - w / 2) * width_right,
                (cy - h / 2) * height_right,
                (cx + w / 2) * width_right,
                (cy + h / 2) * height_right,
            ]
            right_labels += [
                {
                    "bbox_2d": convert_to_qwen25vl_format( # qwen2.5用にbboxをリサイズする
                        bbox_2d, height_right, width_right
                    )
                }
            ]

        return {
            "prompt": messages,
            "images": example["images"],
            "left_labels": left_labels,
            "right_labels": right_labels,
        }

    dataset = load_dataset("Taurine511/spot_diff", split="train")
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    return dataset


def prepare_repeated_dataset():
    dataset = prepare_dataset()
    return Dataset.from_list([dataset[1]] * len(dataset))


def prepare_clever_dataset():
    def format_example(example):
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": None, "type": "image"},
                    {"text": None, "type": "image"},
                    {
                        "text": clever_template.format(
                            instruction=example["instruction"]
                        ),
                        "type": "text",
                    },
                ],
            }
        ]

        images = [example["image1"], example["image2"]]
        return {"prompt": messages, "images": images, "labels": example["sentences"]}

    dataset = load_dataset("Lancelot53/clevr-change", split="validation")
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    return dataset
