import json
import os

from openai import Client

client = Client(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
)


def get_gpt_response(user_prompt):
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ],
    )
    result = response.choices[0].message.content
    if result is None:
        result = "0.0"
    return result


sentence_matching_prompt = """
You are given two **sets of textual descriptions** that describe the changes between two images:

1. A **ground-truth description set**
2. A **model-predicted description set**

Each list item is treated as **one atomic set element**.  
Two elements are considered identical provided they represent almost the same meaning.

Your task is to compute:

- The **size of the intersection** of the ground-truth set and the model-predicted set

### Input

**Ground-truth set:**
{labels}

**Model-predicted set:**
{preds}

### Output

Return **only a single integer** representing the number of exactly matching elements.
""".strip()


def calc_jaccard_index_sentence_match(preds, labels):
    user_prompt = sentence_matching_prompt.format(
        preds=json.dumps(preds, indent=2), labels=json.dumps(labels, indent=2)
    )
    try:
        response = get_gpt_response(user_prompt)
        intersection_size = float(response)
        result = intersection_size / (len(preds) + len(labels) - intersection_size)
    except:
        result = 0.0

    return result


if __name__ == "__main__":
    labels = [
        "the large cyan cylinder changed to shiny",
        "the big cyan matte cylinder that is in front of the green rubber cube turned metal",
        "the big rubber cylinder changed to metal",
        "the big cyan matte object changed to metal",
        "the large cyan rubber cylinder that is to the left of the tiny cyan thing became metal",
        "the large cyan rubber cylinder that is behind the large red rubber block became shiny",
    ]

    preds = [
        "the large cyan cylinder changed to shiny",
        "the big cyan cylinder in front of the green rubber cube turned metal",
        "the big rubber cylinder changed to metal",
        "the green cylinder changed to red",
    ]

    print(f"Jaccard: {calc_jaccard_index_sentence_match(preds, labels)}")
