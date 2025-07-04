import json
import re
from pathlib import Path
import torch


# Used to check our models performance on multiple choice tasks. This can also be done in a more involved way with e.g. LLM-as-a-judge
def check_multiple_choice_with_regex(model_outputs, correct_answers):
    results = []
    for model_output, correct_answer in zip(model_outputs, correct_answers):
        # Strip any trailing newlines and convert to uppercase
        correct_answer = correct_answer.rstrip("\n").upper()

        # Look for the answer letter at the beginning of a line or as the last word
        patterns = [
            rf"\b{correct_answer}\b",  # Word boundary around the answer letter
            rf"\b{correct_answer}[.,)]",  # Answer followed by punctuation
            rf"\(.*{correct_answer}.*\)",  # Answer within parentheses
        ]

        match_found = False
        for pattern in patterns:
            if re.search(pattern, model_output):
                match_found = True
                break  # Exit inner loop once a match is found
        results.append(match_found)
    return results


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
    """
    Apply top-k and/or nucleus (top-p) filtering to logits.
    """
    top_k = min(top_k, logits.size(-1))  # Safety

    if top_k > 0:
        # Remove all tokens with a probability less than the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p

        # Always keep the first token
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


def read_text(file: str) -> str:
    data = ""
    with open(Path(file), "r", encoding="utf-8") as f:
        data = f.read()
    return data


def read_text_lines(file: str) -> list[str]:
    data = []
    with open(Path(file), "r", encoding="utf-8") as f:
        data = f.readlines()
    return data


def read_text_lines_no_empty_line(file: str) -> list[str]:
    data = []
    with open(Path(file), "r", encoding="utf-8") as f:
        data = f.readlines()
    return [line.strip() for line in data if len(line.strip()) > 0]


def save_json(obj: dict, file: str) -> None:
    with open(file, "w", encoding="utf-8") as f:
        json.dump(obj, fp=f, ensure_ascii=False, indent=4)


def read_json(file: str) -> dict | list[dict]:
    data = []
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_to_jsonl(data: list[dict], file: str):
    str_data = [f"{json.dumps(item, ensure_ascii=False)}\n" for item in data]
    with open(file, "w", encoding="utf-8") as f:
        f.writelines(str_data)


def read_jsonl(file: str) -> list[dict]:
    data = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            data.append(json.loads(line))
    return data


def save_text_list(data: list[str], file: str):
    with open(file, "w", encoding="utf-8") as f:
        f.writelines(data)


def save_text(data: str, file: str):
    with open(file, "w", encoding="utf-8") as f:
        f.write(data)
