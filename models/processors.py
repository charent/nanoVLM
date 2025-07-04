from transformers import AutoTokenizer
import torchvision.transforms as transforms

TOKENIZERS_CACHE = {}


def get_tokenizer(name, extra_special_tokens=None, chat_template=None):
    if name not in TOKENIZERS_CACHE:
        tokenizer_init_kwargs = {"use_fast": True}
        if extra_special_tokens is not None:
            tokenizer_init_kwargs["extra_special_tokens"] = extra_special_tokens
        if chat_template is not None:
            tokenizer_init_kwargs["chat_template"] = chat_template
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            **tokenizer_init_kwargs,
        )
        tokenizer.pad_token = tokenizer.eos_token
        TOKENIZERS_CACHE[name] = tokenizer
    return TOKENIZERS_CACHE[name]


def get_image_processor(vm_input_imgage_size: tuple):
    return transforms.Compose(
        [transforms.Resize(vm_input_imgage_size), transforms.ToTensor()]
    )


def vlm_input_apply_chat_template(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str | list]],
    tokenize: bool = True,
    add_generation_prompt: bool = True,
    mp_image_token_length: int = 64,
    return_tensors: str = "pt",
    enable_thinking: bool = False,
    **tokenize_kwargs
):
    text: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )

    image_token = tokenizer.image_token
    while image_token in text:
        text = text.replace(image_token, "<|placeholder|>" * mp_image_token_length, 1)
    text = text.replace("<|placeholder|>", image_token)

    if tokenize:
        return tokenizer(text, return_tensors=return_tensors, **tokenize_kwargs)
    return text


def vlm_input_apply_chat_template_for_train(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str | list]],
    tokenize: bool = True,
    mp_image_token_length: int = 64,
    return_tensors: str = "pt",
    enable_thinking: bool = False,
    **tokenize_kwargs
):
    text: str = tokenizer.apply_chat_template(
        messages[0:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    text += messages[-1]["content"]

    image_token = tokenizer.image_token
    while image_token in text:
        text = text.replace(image_token, "<|placeholder|>" * mp_image_token_length, 1)
    text = text.replace("<|placeholder|>", image_token)

    if tokenize:
        return tokenizer(text, return_tensors=return_tensors, **tokenize_kwargs)
    return text
