from transformers import Qwen3Config, SiglipVisionConfig, PretrainedConfig
from dataclasses import dataclass, field, asdict
from models.utils import read_text


@dataclass
class LMConfig(Qwen3Config):
    def __init__(
        self,
        vocab_size: int = 151936,
        max_position_embeddings: int = 40960,
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        use_sliding_window: bool = False,
        max_window_layers: int = 28,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-06,
        use_cache: bool = True,
        rope_theta: int = 1000000,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        tie_word_embeddings: bool = True,
        chunk_size_feed_forward: int = 0,
        is_encoder_decoder: bool = False,
        is_decoder: bool = False,
        add_cross_attention: bool = False,
        tie_encoder_decoder: bool = False,
        bos_token_id: int = 151643,
        eos_token_id: int = 151645,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size: int = vocab_size
        self.max_position_embeddings: int = max_position_embeddings
        self.hidden_size: int = hidden_size
        self.intermediate_size: int = intermediate_size
        self.num_hidden_layers: int = num_hidden_layers
        self.num_attention_heads: int = num_attention_heads
        self.use_sliding_window: bool = use_sliding_window
        self.max_window_layers: int = max_window_layers
        self.num_key_value_heads: int = num_key_value_heads
        self.head_dim: int = head_dim
        self.hidden_act: str = hidden_act
        self.initializer_range: float = initializer_range
        self.rms_norm_eps: float = rms_norm_eps
        self.use_cache: bool = use_cache
        self.rope_theta: int = rope_theta
        self.attention_bias: bool = attention_bias
        self.attention_dropout: float = attention_dropout
        self.tie_word_embeddings: bool = tie_word_embeddings
        self.chunk_size_feed_forward: int = chunk_size_feed_forward
        self.is_encoder_decoder: bool = is_encoder_decoder
        self.is_decoder: bool = is_decoder
        self.add_cross_attention: bool = add_cross_attention
        self.tie_encoder_decoder: bool = tie_encoder_decoder
        self.bos_token_id: int = bos_token_id
        self.eos_token_id: int = eos_token_id


@dataclass
class VMConfig(SiglipVisionConfig):
    def __init__(
        self,
        tie_word_embeddings: bool = True,
        chunk_size_feed_forward: int = 0,
        is_encoder_decoder: bool = False,
        is_decoder: bool = False,
        model_type: str = "siglip_vision_model",
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        patch_size: int = 16,
        image_size: int = 256,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-06,
        hidden_act: str = "gelu_pytorch_tanh",
        output_hidden_states: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.tie_word_embeddings: bool = tie_word_embeddings
        self.chunk_size_feed_forward: int = chunk_size_feed_forward
        self.is_encoder_decoder: bool = is_encoder_decoder
        self.is_decoder: bool = is_decoder
        self.model_type: str = model_type
        self.hidden_size: int = hidden_size
        self.intermediate_size: int = intermediate_size
        self.num_hidden_layers: int = num_hidden_layers
        self.num_attention_heads: int = num_attention_heads
        self.num_channels: int = num_channels
        self.patch_size: int = patch_size
        self.image_size: int = image_size
        self.attention_dropout: float = attention_dropout
        self.layer_norm_eps: float = layer_norm_eps
        self.hidden_act: str = hidden_act
        self.output_hidden_states: bool = output_hidden_states


@dataclass
class MPConfig(PretrainedConfig):
    def __init__(
        self,
        mp_pixel_shuffle_factor: int = 2,
        mp_image_token_length: int = 64,
        mp_projector_up_factor: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.mp_pixel_shuffle_factor: int = mp_pixel_shuffle_factor
        self.mp_image_token_length: int = mp_image_token_length
        self.mp_projector_up_factor: int = mp_projector_up_factor


@dataclass
class VLMConfig(PretrainedConfig):
    model_type = "qwen3"

    def __init__(
        self,
        language_model_config: LMConfig = None,
        vision_model_config: VMConfig = None,
        modality_model_config: MPConfig = None,
        vlm_extra_tokens: dict[str, str] = None,
        lm_chat_template: str = None,
        vm_input_image_size: tuple = (256, 256),
        vm_image_token_length: int = 64,
        vm_pretrain_path: str = "model_save/google/siglip2-base-patch16-256",
        lm_pretrain_path: str = "model_save/Qwen/Qwen3-0___6B",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.language_model_config: LMConfig = (
            language_model_config if language_model_config is not None else LMConfig()
        )
        self.vision_model_config: VMConfig = (
            vision_model_config if vision_model_config is not None else VMConfig()
        )
        self.modality_model_config: MPConfig = (
            modality_model_config if modality_model_config is not None else MPConfig()
        )
        self.vlm_extra_tokens: dict[str, str] = (
            vlm_extra_tokens
            if vlm_extra_tokens is not None
            else {
                "image_token": "<|image_pad|>",
                "boi_token": "<|vision_start|>",
                "eoi_token": "<|vision_end|>",
            }
        )
        self.lm_chat_template: str = (
            lm_chat_template
            if lm_chat_template is not None
            else read_text(r"models/chat_template.jinja")
        )
        self.vm_input_image_size: tuple = vm_input_image_size
        self.vm_image_token_length: int = vm_image_token_length
        self.vm_pretrain_path: str = vm_pretrain_path
        self.lm_pretrain_path: str = lm_pretrain_path
