"""
MiMo Audio 模型配置类
"""

from dataclasses import dataclass
from typing import Optional, List, TYPE_CHECKING
from bonsai.models.qwen3.modeling import ShardingCfg

if TYPE_CHECKING:
    from bonsai.models.qwen2.modeling import ModelConfig as Qwen2Config


@dataclass
class MiMoAudioConfig:
    """Configuration for MiMo Audio Model"""

    vocab_size: int = 151936
    hidden_size: int = 1536
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    intermediate_size: int = 8960
    max_position_embeddings: int = 32768
    rope_theta: int = 10000  # RoPE 基础频率

    # MiMo-specific config
    speech_vocab_size: str | int = "1025-1025-129-129-129-129-129-129"
    speech_zeroemb_idx: str | int = "1024-1024-128-128-128-128-128-128"
    delay_pattern: str = "0-1-2-3-4-5-6-7"
    head_dim: int = 128
    group_size: int = 4
    audio_channels: int = 8

    # Local transformer config
    local_dim: int = 1024
    local_layers: int = 16
    local_attn_heads: int = 64
    local_ffn_dim: int = 4096
    local_attn_dropout: float = 0.1

    # Input local transformer config
    input_local_layers: int = 6
    input_local_dim: Optional[int] = None
    input_full_attention: Optional[bool] = None

    # Sharding config
    shd_cfg: ShardingCfg = ShardingCfg.no_sharding()

    def __post_init__(self):
        if self.input_local_dim is None:
            self.input_local_dim = self.local_dim

    @classmethod
    def with_sharding(cls, **kwargs):
        """Create config with default sharding enabled"""
        kwargs['shd_cfg'] = ShardingCfg.default()
        return cls(**kwargs)

    def _parse_maybe_list(self, value: str | int, length: int) -> List[int]:
        if isinstance(value, str) and "-" in value:
            return [int(s) for s in value.split("-")]
        return [int(value)] * length

    def parsed_speech_empty_ids(self) -> List[int]:
        return self._parse_maybe_list(self.speech_zeroemb_idx, self.audio_channels)

    def parsed_speech_vocab_sizes(self) -> List[int]:
        return self._parse_maybe_list(self.speech_vocab_size, self.audio_channels)

    def parsed_delay_pattern(self) -> List[int]:
        return self._parse_maybe_list(self.delay_pattern, self.audio_channels)

    def create_qwen2_config(self) -> "Qwen2Config":
        """Create Qwen2 config for main transformer from MiMo config"""
        from bonsai.models.qwen2.modeling import ModelConfig as Qwen2Config

        return Qwen2Config(
            num_layers=self.num_hidden_layers,
            vocab_size=self.vocab_size,
            emb_dim=self.hidden_size,
            mlp_dim=self.intermediate_size,
            num_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_key_value_heads,
            rope_theta=self.rope_theta,
            norm_eps=1e-6,
            tie_word_embeddings=False,
            shd_cfg=self.shd_cfg,
        )

    def create_local_qwen2_config(self) -> "Qwen2Config":
        """Create local transformer Qwen2 config"""
        from bonsai.models.qwen2.modeling import ModelConfig as Qwen2Config

        return Qwen2Config(
            num_layers=self.local_layers,
            vocab_size=self.vocab_size,
            emb_dim=self.local_dim,
            mlp_dim=self.local_ffn_dim,
            num_heads=self.local_attn_heads,
            head_dim=self.local_dim // self.local_attn_heads,
            num_kv_heads=self.local_attn_heads,
            rope_theta=self.rope_theta,
            norm_eps=1e-6,
            tie_word_embeddings=False,
            shd_cfg=self.shd_cfg,
        )

    def create_input_local_qwen2_config(self) -> "Qwen2Config":
        """Create input local transformer Qwen2 config"""
        from bonsai.models.qwen2.modeling import ModelConfig as Qwen2Config

        # Convert input_full_attention to use_causal_mask
        # input_full_attention=True -> use_causal_mask=False (bidirectional)
        # input_full_attention=False/None -> use_causal_mask=True (causal, backward compatible)
        use_causal_mask = not self.input_full_attention if self.input_full_attention is not None else True

        return Qwen2Config(
            num_layers=self.input_local_layers,
            vocab_size=self.vocab_size,
            emb_dim=self.input_local_dim,
            mlp_dim=self.input_local_dim * 4,
            num_heads=self.local_attn_heads,
            head_dim=self.input_local_dim // self.local_attn_heads,
            num_kv_heads=self.local_attn_heads,
            rope_theta=self.rope_theta,
            norm_eps=1e-6,
            tie_word_embeddings=False,
            use_causal_mask=use_causal_mask,
            shd_cfg=self.shd_cfg,
        )


@dataclass
class MiMoAudioArguments:
    """Arguments for special token indices"""
    model_name_or_path: str
    sosp_idx: int
    eosp_idx: int
    sostm_idx: int
    eostm_idx: int
    eot_idx: int
    empty_idx: int


@dataclass
class MiMoSamplerConfig:
    """Sampler configuration for text/audio generation"""
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
