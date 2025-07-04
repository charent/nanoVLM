import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

from transformers import (
    Qwen3ForCausalLM,
    SiglipVisionModel,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.modeling_utils import unwrap_model
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)

from models.config import MPConfig, LMConfig, VMConfig, VLMConfig
from models.processors import get_tokenizer
from models.modality_projector import ModalityProjector
from models.utils import top_k_top_p_filtering


class VisionLanguageModel(PreTrainedModel):
    def __init__(self, config: VLMConfig, load_from_pretrained=True, **kwargs):
        super().__init__(config, **kwargs)

        self.vision_encoder: Qwen3ForCausalLM
        self.decoder: Qwen3ForCausalLM
        if load_from_pretrained:
            print("Loading weights from pretrained")
            self.vision_encoder = SiglipVisionModel.from_pretrained(
                config.vm_pretrain_path
            )
            self.decoder = Qwen3ForCausalLM.from_pretrained(config.lm_pretrain_path)
        else:
            self.vision_encoder = SiglipVisionModel(config.vision_model_config)
            self.decoder = Qwen3ForCausalLM(config.language_model_config)

        self.MP = ModalityProjector(config)
        self.load_from_pretrained = load_from_pretrained

        self.tokenizer = get_tokenizer(
            config.lm_pretrain_path,
            config.vlm_extra_tokens,
            config.lm_chat_template,
        )

        config_class = VLMConfig
        self.config = config

    def get_input_embeddings(self):
        return self.decoder.model.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.decoder.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.decoder.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.decoder.model = decoder

    def get_decoder(self):
        return self.decoder.model

    def _replace_img_tokens_with_embd(
        self,
        input_ids: torch.Tensor,
        token_embd: torch.Tensor,
        image_embd: torch.Tensor,
    ):
        """
        Replace every image-token placeholder in `input_ids` with the corresponding slice
        from `image_embd`. Supports an arbitrary number of image-token placeholders per sample.
        The first example in the batch might have 2 images and the second none.
        """
        # Clone the original embeddings to avoid in-place issues
        updated_token_embd = token_embd.clone()

        # Build a mask of all image-token positions: shape [B, T_seq]
        mask = input_ids == self.tokenizer.image_token_id
        updated_token_embd[mask] = image_embd.view(-1, image_embd.size(-1)).to(
            updated_token_embd.dtype
        )  # torch flattens before assigning

        return updated_token_embd

    def forward(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        if isinstance(images, list) and isinstance(
            images[0], list
        ):  # If images is a list of lists, flatten it
            images = [img for sublist in images for img in sublist]
            images = torch.stack(images).to(input_ids.device)

        image_embd: BaseModelOutputWithPooling = self.vision_encoder(images)

        image_embd = self.MP(
            image_embd.last_hidden_state
        )  # [num_images, mp_image_token_length, D_lm]

        token_embd = self.decoder.model.embed_tokens(input_ids)  # [B, T_sequence, D_lm]

        updated_token_embd = self._replace_img_tokens_with_embd(
            input_ids, token_embd, image_embd
        )

        # The updated_token_embd is now the token_embd with image parts replaced.
        # The attention_mask comes from the collator and should already cover the full sequence.
        decoder_outputs: CausalLMOutputWithPast = self.decoder(
            inputs_embeds=updated_token_embd, attention_mask=attention_mask
        )
        logits = decoder_outputs.logits
        loss = None
        if labels is not None:

            loss = ForCausalLMLoss(
                logits=logits,
                labels=labels,
                vocab_size=logits.size(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor | list,
        attention_mask: torch.Tensor = None,
        max_new_tokens: int = 5,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 0.5,
        greedy: bool = False,
    ):
        if isinstance(images, list) and isinstance(
            images[0], list
        ):  # If images is a list of lists, flatten it
            images = [img for sublist in images for img in sublist]
            images = torch.stack(images).to(input_ids.device)

        # 1. Process image
        image_embd: BaseModelOutputWithPooling = self.vision_encoder(
            images
        )  # [B, T_img_feat, D_model]

        image_embd = self.MP(
            image_embd.last_hidden_state
        )  # [B, mp_image_token_length, D_lm]

        # 2. Embed initial text prompt tokens
        prompt_token_embeds = self.decoder.model.embed_tokens(
            input_ids
        )  # [B, T_prompt_text, D_lm]

        # 3. Combine image and text embeddings
        initial_combined_embeds = self._replace_img_tokens_with_embd(
            input_ids, prompt_token_embeds, image_embd
        )

        current_total_seq_len = initial_combined_embeds.size(1)
        batch_size = input_ids.size(0)  # Or initial_combined_embeds.size(0)

        # --- Multimodal Prefill Phase ---
        # prefill_output, kv_cache_list = self.decoder(
        decoder_outputs: CausalLMOutputWithPast = self.decoder(
            inputs_embeds=initial_combined_embeds,
            attention_mask=attention_mask,  # Use the provided attention mask
            use_cache=True,
        )
        prefill_output = decoder_outputs.logits
        past_key_values = decoder_outputs.past_key_values

        last_token_output_from_prefill = prefill_output[:, -1, :]
        current_logits = last_token_output_from_prefill

        # Store newly generated token IDs
        newly_generated_ids_list = []

        # --- Decode Phase by sampling tokens autoregressively using the kv-cache ---
        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(
                    current_logits, top_k=top_k, top_p=top_p
                )
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            newly_generated_ids_list.append(next_token_id)

            # Embed the newly generated token
            next_token_embed = self.decoder.model.embed_tokens(
                next_token_id
            )  # [B, 1, D_lm]

            current_total_seq_len += 1

            # update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (batch_size, 1),
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        ),
                    ),
                    dim=1,
                )

            # With KV cache: only process the new token
            decoder_outputs: CausalLMOutputWithPast = self.decoder(
                inputs_embeds=next_token_embed,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = decoder_outputs.past_key_values
            current_logits = decoder_outputs.logits[:, -1, :]

        if not newly_generated_ids_list:  # Handle case where max_new_tokens might be 0
            return torch.empty(
                (batch_size, 0), dtype=torch.long, device=input_ids.device
            )

        generated_ids = torch.cat(newly_generated_ids_list, dim=1)

        # Post-process to handle EOS token.
        if (
            self.tokenizer.eos_token_id is not None and generated_ids.numel() > 0
        ):  # Ensure generated_ids is not empty
            seq_len = generated_ids.size(1)
            device = generated_ids.device

            eos_mask = (
                generated_ids == self.tokenizer.eos_token_id
            )  # Create a boolean mask for EOS tokens

            col_indices_for_min = torch.arange(
                seq_len, device=device
            )  # Create column indices [0, 1, ..., seq_len-1]

            # In eos_mask, mark positions with actual col_idx, others with a large number
            masked_col_indices = torch.where(
                eos_mask,
                col_indices_for_min.unsqueeze(0).expand_as(generated_ids),
                seq_len + 1,
            )

            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values

            # Clamp values to seq_len (if no EOS found, min will be seq_len + 1, clamp brings it to seq_len0. This means if no EOS, or EOS is the last token, no replacement will happen for that sample.
            actual_first_eos_indices = torch.clamp(
                first_eos_indices_values, max=seq_len
            )

            # Create column indices for comparison, shape [batch_size, seq_len]
            col_indices_for_comparison = (
                torch.arange(seq_len, device=device)
                .unsqueeze(0)
                .expand_as(generated_ids)
            )

            # Tokens are replaced if their column index is greater than the index of the first EOS token
            replace_mask = (
                col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)
            )

            generated_ids[replace_mask] = self.tokenizer.eos_token_id

        return generated_ids
