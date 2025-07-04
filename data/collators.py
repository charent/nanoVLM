import torch


class BaseCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _pad_batch(self, batch, max_length):
        """ """
        # padding_side = 'right' for training
        batch["input_ids"] = [
            torch.nn.functional.pad(
                torch.tensor(ids),
                (0, max_length - len(ids)),
                value=self.tokenizer.pad_token_id,
            )
            for ids in batch["input_ids"]
        ]
        batch["labels"] = [
            torch.nn.functional.pad(
                torch.tensor(labels),
                (0, max_length - len(labels)),
                value=-100,
            )
            for labels in batch["labels"]
        ]
        batch["attention_mask"] = [
            torch.nn.functional.pad(
                torch.tensor(attention_mask),
                (0, max_length - len(attention_mask)),
                value=0,
            )
            for attention_mask in batch["attention_mask"]
        ]

    def prepare_batch(self, batch, max_length=None):
        # batch is a list of dicts, each containing "input_ids", "labels", "images"
        # let's convert it to a dict of lists of tensors
        batch = {k: [item[k] for item in batch] for k in batch[0]}

        if max_length is not None:
            batch = self._discard_samples_that_are_too_long(batch, max_length)

        inputs_lengths = list(map(len, batch["input_ids"]))
        # Pad samples to max length
        if max_length is not None:
            max_len = max_length
        else:
            max_len = max(inputs_lengths)

        batch["attention_mask"] = [[1] * length for length in inputs_lengths]

        self._pad_batch(
            batch, max_len
        )  #  dictionaries in Python are mutable and passed by reference

        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "images": torch.stack(batch["images"]),  # only ont images
            "labels": torch.stack(batch["labels"]),
        }

    def _discard_samples_that_are_too_long(self, batch, max_length):
        filtered = [
            (ids, label, img)
            for ids, label, attn, img in zip(
                batch["input_ids"],
                batch["labels"],
                batch["images"],
            )
            if len(ids) <= max_length
        ]
        if not filtered:
            return [], [], [], []
        batch_token_ids, batch_labels, batch_images = zip(*filtered)
        return {
            "input_ids": list(batch_token_ids),
            "labels": list(batch_labels),
            "images": list(batch_images),
        }


class VQACollator(BaseCollator):  # Visual Question Answering Collator
    def __init__(self, tokenizer, max_length):
        self.max_length = max_length
        super().__init__(tokenizer)

    def _pad_batch(
        self, batch, max_length
    ):  # Reimplementing to use -100 as the pad value for labels, so that it's ignored by the loss
        # padding_side = 'right' for training
        batch["input_ids"] = [
            torch.nn.functional.pad(
                torch.tensor(ids),
                (0, max_length - len(ids)),
                value=self.tokenizer.pad_token_id,
            )
            for ids in batch["input_ids"]
        ]
        batch["labels"] = [
            torch.nn.functional.pad(
                torch.tensor(labels),
                (0, max_length - len(labels)),
                value=-100,
            )
            for labels in batch["labels"]
        ]
        batch["attention_mask"] = [
            torch.nn.functional.pad(
                torch.tensor(attention_mask),
                (0, max_length - len(attention_mask)),
                value=0,
            )
            for attention_mask in batch["attention_mask"]
        ]

    def __call__(self, batch):
        batch = self.prepare_batch(batch)
        return batch


class MMStarCollator(BaseCollator):  # https://huggingface.co/datasets/Lin-Chen/MMStar
    def __call__(self, batch):
        batch = self.prepare_batch(batch)
        return batch
