from datasets import (
    Features,
    Value,
    ClassLabel,
    Sequence,
    Image,
    load_dataset,
    concatenate_datasets,
)
from transformers import AutoTokenizer
from PIL import Image as PILImage
from io import BytesIO
from typing import Any

from models.config import VLMConfig
from data.datasets import (
    vlm_input_apply_chat_templat_for_training,
    convert_to_chat_meessages,
)
from models.processors import get_tokenizer, get_image_processor


def process_data(
    samples: dict[str, Any],
    tokenizer: AutoTokenizer,
    mp_image_token_length: int = 64,
    enable_thinking: bool = False,
):

    all_inputs_labels, all_images = [], []

    for image, conversations in zip(samples["image"], samples["conversations"]):
        messages = []
        for item in conversations:
            role = ""
            content: str = item.get("value", "")
            if item["from"] == "human":
                role = "user"
            else:
                role = "assistant"
            messages.append({"role": role, "content": content})

        # print(messages)
        img_bytes = image["bytes"]
        img_stream = BytesIO(img_bytes)
        img = PILImage.open(img_stream)
        img = img.resize((256, 256))

        if img.mode != "RGB":
            img = img.convert("RGB")

        input_ids, labels = vlm_input_apply_chat_templat_for_training(
            tokenizer, messages, mp_image_token_length, enable_thinking=enable_thinking
        )
        all_inputs_labels.append(
            {
                "input_ids": input_ids,
                "labels": labels,
            }
        )
        all_images.append([img])

    return {"inputs": all_inputs_labels, "images": all_images}


# features = Features(
#     {
#         "messages": Sequence(
#             Features(
#                 {
#                     "role": Value("string"),
#                     "content": Value("string"),
#                 }
#             )
#         ),
#         "images": Sequence(
#             Features(
#                 {
#                     "image": Image(),
#                     "index": Value("int16"),
#                 }
#             )
#         ),
#     }
# )


if __name__ == "__main__":

    config = VLMConfig()

    tokenizer = get_tokenizer(
        config.lm_pretrain_path, config.vlm_extra_tokens, config.lm_chat_template
    )
    image_processor = get_image_processor(config.vm_input_image_size)
    data_content_image_pad_str = "<image>\n"

    root_dir = r'E:\Cache\hf_cache\hub\datasets--Emova-ollm--emova-alignment-7m\snapshots\03f2fd4d001cc38824d86a110b46a938023c19a4\allava-caption-zh-part2'
    
    datasets = []
    for i in range(15):
        ds  = load_dataset(
            "parquet", data_files={"train": rf"{root_dir}/train-000{i:02}-of-00024.parquet"}
        )["train"]
        datasets.append(ds)

    raw_data = concatenate_datasets(datasets)
    map_fun_args = {
        "tokenizer": tokenizer,
        "mp_image_token_length": config.vm_image_token_length,
    }
    maped_dataste = raw_data.map(
        process_data,
        batched=True,
        batch_size=2048,
        remove_columns=raw_data.column_names,
        num_proc=8,
        fn_kwargs=map_fun_args,
        # features=features,
    )

    save_dir = "dataset_dir"
    print(f"save dataset with size: {len(maped_dataste)} to: {save_dir}")

    maped_dataste.save_to_disk(save_dir)
