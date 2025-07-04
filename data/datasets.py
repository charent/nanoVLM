from typing import Any
import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_from_disk
from transformers import AutoTokenizer

from models.processors import get_tokenizer, get_image_processor
from models.config import VLMConfig


def convert_to_chat_meessages(
    messages: list[dict, str], image_pad_str: str = "<image>\n"
):
    """
    转换为openai对话模板以应用chat template
    """
    messages = messages.copy()
    for msg in messages:
        if msg["role"] == "user":
            content: str = msg["content"]
            img_index = content.find(image_pad_str)
            img_cnt = content.count(image_pad_str)
            content = content.replace(image_pad_str, "")
            contents = []
            if img_index == 0:
                contents.extend([{"type:": "image", "image": ""}] * img_cnt)
                contents.append({"type:": "text", "text": content})
            elif img_cnt > 0:
                contents.append({"type:": "text", "text": content})
                contents.extend([{"type:": "image", "image": ""}] * img_cnt)

            msg["content"] = contents
    return messages


def vlm_input_apply_chat_templat_for_training(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str | list]],
    mp_image_token_length: int = 64,
    enable_thinking: bool = False,
):

    messages = convert_to_chat_meessages(messages)
    text: str = tokenizer.apply_chat_template(
        messages[0:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    image_token = tokenizer.image_token
    while image_token in text:
        text = text.replace(image_token, "<|placeholder|>" * mp_image_token_length, 1)
    text = text.replace("<|placeholder|>", image_token)

    prompt_tokens = tokenizer(text, return_attention_mask=False)["input_ids"]
    response_tokens = tokenizer(
        messages[-1]["content"],
        add_special_tokens=False,
    )["input_ids"]

    eos_token_id = tokenizer.eos_token_id
    input_ids = prompt_tokens + response_tokens + [eos_token_id]
    labels = [-100] * len(prompt_tokens) + response_tokens + [eos_token_id]

    return input_ids, labels


class VQADataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        config: VLMConfig,
        train_max_token_length: int = 1024,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = get_tokenizer(
            config.lm_pretrain_path, config.vlm_extra_tokens, config.lm_chat_template
        )
        self.image_processor = get_image_processor(config.vm_input_image_size)
        self.max_length = train_max_token_length
        self.imgage_token_length = config.vm_image_token_length

        self.dataset = load_from_disk(dataset_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """ """

        row = self.dataset[index]
        inputs = row["inputs"]
        image = row["images"][0]
        input_ids, labels = inputs["input_ids"], inputs["labels"]

        if len(input_ids) > self.max_length:
            input_ids = input_ids[0 : self.max_length]
            labels = labels[0 : self.max_length]

        images = self.image_processor(image)

        return dict(
            input_ids=input_ids,
            labels=labels,
            images=images,
        )
