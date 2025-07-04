# Modality Projection from Vision to Language
import torch.nn as nn
from models.config import VLMConfig
import torch
from transformers import PreTrainedModel


class ModalityProjector(PreTrainedModel):
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.config = config
        self.input_dim = config.vision_model_config.hidden_size * (
            config.modality_model_config.mp_pixel_shuffle_factor**2
        )
        self.output_dim = config.language_model_config.hidden_size
        self.scale_factor = config.modality_model_config.mp_pixel_shuffle_factor
        self.mp_projector_up_factor = (
            config.modality_model_config.mp_projector_up_factor
        )

        self.up_proj = nn.Linear(
            self.input_dim, self.input_dim * self.mp_projector_up_factor, bias=False
        )
        self.down_proj = nn.Linear(
            self.input_dim * self.mp_projector_up_factor, self.output_dim, bias=False
        )
        self.act_func = nn.SiLU()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
    def pixel_shuffle(self, x: torch.Tensor):
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq**0.5)
        assert (
            seq_root**2 == seq
        )  # Sequence length must be a perfect square for pixel shuffle
        assert (
            seq_root % self.scale_factor == 0
        )  # Sequence root must be divisible by scale factor

        height = width = seq_root
        x = x.view(bsz, height, width, embed_dim)
        h_out = height // self.scale_factor
        w_out = width // self.scale_factor

        x = x.reshape(
            bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor**2)

        return x

    def forward(self, vision_hidden_states: torch.Tensor):
        vision_hidden_states = self.pixel_shuffle(vision_hidden_states)

        outputs = self.down_proj(self.act_func(self.up_proj(vision_hidden_states)))

        return outputs
