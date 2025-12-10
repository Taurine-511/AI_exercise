from datasets import load_dataset

user_prompt = """
You are given two images: a reference image (Image A) and a modified image (Image B).
Identify all visual differences between Image A and Image B.
Output each detected target's bbox coordinates in JSON format. The format of the bbox coordinates is:

[
    "bbox_2d": [x1, y1, x2, y2], ”label”: ”target name”,
    "bbox_2d": [x1, y1, x2, y2], ”label”: ”target name”,
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

    dataset = dataset.add_column("prompt", [messages] * len(dataset))

    return dataset
