import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple
import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig
from jax.sharding import PartitionSpec as P, get_abstract_mesh, reshard

Array = jnp.ndarray
ShardingSpec = P


def shard(x: Array, s: ShardingSpec) -> Array:
    """Apply sharding to array if mesh is available."""
    mesh = get_abstract_mesh()
    if not mesh.empty and len(mesh.axis_names) > 0:
        return reshard(x, s)
    return x


@dataclass(slots=True, frozen=True)
class MiMoShardingCfg:
    """Sharding configuration for MiMo Audio Tokenizer.

    Controls how model parameters and activations are distributed across devices.
    """
    # Conv层权重 sharding
    conv_weight: ShardingSpec  # (in_channels, out_channels, kernel_size)
    conv_bias: ShardingSpec    # (out_channels,)

    # Transformer权重 sharding (Encoder/Decoder/Vocoder共用)
    attn_qkvo_weight: ShardingSpec  # (d_model, d_model)
    attn_qkv_bias: ShardingSpec     # (d_model,)
    attn_out_bias: ShardingSpec     # (d_model,)

    # FFN权重 sharding
    ffn_weight_in: ShardingSpec   # (d_model, ffn_dim)
    ffn_weight_out: ShardingSpec  # (ffn_dim, d_model)
    ffn_bias: ShardingSpec        # (ffn_dim,) or (d_model,)

    # LayerNorm/GroupNorm sharding
    norm_scale: ShardingSpec      # (dim,)
    norm_bias: ShardingSpec       # (dim,)

    # Quantizer codebook sharding
    codebook: ShardingSpec        # (codebook_size, d_model)

    # ConvTranspose1d权重 sharding
    conv_transpose_weight: ShardingSpec  # (in_ch, out_ch, kernel)
    conv_transpose_bias: ShardingSpec    # (out_ch,)

    # ISTFT相关 sharding
    istft_linear_weight: ShardingSpec  # (dim, n_fft+2)
    istft_linear_bias: ShardingSpec    # (n_fft+2,)
    istft_window: ShardingSpec         # (win_length,)

    # 激活值 sharding
    act_btd: ShardingSpec         # [batch, time, d_model]
    act_btnh: ShardingSpec        # [batch, time, num_heads, head_dim]
    act_btc: ShardingSpec         # [batch, time, channels]

    @staticmethod
    def no_sharding():
        """Configuration with no sharding (all None)."""
        return MiMoShardingCfg(
            conv_weight=P(None, None, None),
            conv_bias=P(None),
            attn_qkvo_weight=P(None, None),
            attn_qkv_bias=P(None),
            attn_out_bias=P(None),
            ffn_weight_in=P(None, None),
            ffn_weight_out=P(None, None),
            ffn_bias=P(None),
            norm_scale=P(None),
            norm_bias=P(None),
            codebook=P(None, None),
            conv_transpose_weight=P(None, None, None),
            conv_transpose_bias=P(None),
            istft_linear_weight=P(None, None),
            istft_linear_bias=P(None),
            istft_window=P(None),
            act_btd=P(None, None, None),
            act_btnh=P(None, None, None, None),
            act_btc=P(None, None, None),
        )

    @staticmethod
    def default():
        """Default sharding configuration for distributed training."""
        return MiMoShardingCfg(
            conv_weight=P(None, "tp", None),
            conv_bias=P("tp"),
            attn_qkvo_weight=P("fsdp", "tp"),
            attn_qkv_bias=P("tp"),
            attn_out_bias=P("tp"),
            ffn_weight_in=P("fsdp", "tp"),
            ffn_weight_out=P("tp", "fsdp"),
            ffn_bias=P("tp"),
            norm_scale=P("tp"),
            norm_bias=P("tp"),
            codebook=P("tp", "fsdp"),
            conv_transpose_weight=P(None, "tp", None),
            conv_transpose_bias=P("tp"),
            istft_linear_weight=P("fsdp", "tp"),
            istft_linear_bias=P("tp"),
            istft_window=P(None),  # replicated
            act_btd=P("fsdp", None, "tp"),
            act_btnh=P("fsdp", None, "tp", None),
            act_btc=P("fsdp", None, "tp"),
        )


class MiMoAudioTokenizerConfig(PretrainedConfig):
    model_type = "mimo_audio_tokenizer"

    def __init__(
            self,
            max_audio_seconds: int = 1800,
            stride_size: int = 2,
            avg_pooler: int = 2,
            d_model: int = 1280,
            scale_embedding: bool = False,
            kernel_size: int = 3,
            activation_function: str = "gelu",
            encoder_layers: int = 32,
            encoder_skip_layer_id: int = 3,
            encoder_attention_heads: int = 20,
            encoder_ffn_dim: int = 5120,
            encoder_causal: bool = False,
            encoder_attn_window_size: list[int] = None,  # [-1,-1]
            decoder_layers: int = 32,
            decoder_attention_heads: int = 20,
            decoder_ffn_dim: int = 5120,
            decoder_kernel_size: int = 3,
            decoder_stride_size: int = 2,
            decoder_causal: bool = True,
            decoder_attn_window_size: list[int] = None,  # [-1,-1]
            nfft: int = 960,
            vocoder_dim: int = 256,
            vocoder_intermediate_dim: int = 1024,
            vocoder_num_layers: int = 16,
            n_mels: int = 128,
            sampling_rate: int = 24000,
            hop_length: int = 240,
            window_size: int = 960,
            vocoder_padding: str = "same",
            fmin: int = 0,
            fmax: int = None,
            num_quantizers: int = 20,
            codebook_size: list[int] = None,
            # [1024,1024,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128]
            threshold_ema_dead_code: int = 2,
            position_embedding_type: str = "rope",
            rope_theta: int = 10000,
            rope_type: str = "default",
            ln_type: str = "LayerNorm",
            vocoder_attention_heads: int = 16,
            vocoder_attn_window_size: list[int] = None,  # [40,10]
            use_sharding: bool = False,  # 新增
            shd_cfg: MiMoShardingCfg | None = None,  # 新增
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_audio_seconds = max_audio_seconds
        self.stride_size = stride_size
        self.avg_pooler = avg_pooler
        self.d_model = d_model
        self.scale_embedding = scale_embedding
        self.kernel_size = kernel_size
        self.activation_function = activation_function
        self.encoder_layers = encoder_layers
        self.encoder_skip_layer_id = encoder_skip_layer_id
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_causal = encoder_causal
        self.encoder_attn_window_size = (
            encoder_attn_window_size
            if encoder_attn_window_size is not None
            else [-1, -1]
        )
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_kernel_size = decoder_kernel_size
        self.decoder_stride_size = decoder_stride_size
        self.decoder_causal = decoder_causal
        self.decoder_attn_window_size = (
            decoder_attn_window_size
            if decoder_attn_window_size is not None
            else [-1, -1]
        )
        self.nfft = nfft
        self.vocoder_dim = vocoder_dim
        self.vocoder_intermediate_dim = vocoder_intermediate_dim
        self.vocoder_num_layers = vocoder_num_layers
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.window_size = window_size
        self.vocoder_padding = vocoder_padding
        self.fmin = fmin
        self.fmax = fmax
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size if codebook_size is not None else [1024]
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.position_embedding_type = position_embedding_type
        self.rope_theta = rope_theta
        self.rope_type = rope_type
        self.ln_type = ln_type
        self.vocoder_attention_heads = vocoder_attention_heads
        self.vocoder_attn_window_size = (
            vocoder_attn_window_size
            if vocoder_attn_window_size is not None
            else [40, 10]
        )

        # Sharding configuration
        if shd_cfg is None:
            self.shd_cfg = MiMoShardingCfg.default() if use_sharding else MiMoShardingCfg.no_sharding()
        else:
            self.shd_cfg = shd_cfg


def make_sequence_mask(lengths: Array, max_length: Optional[int] = None) -> Array:
    max_len = max_length or int(jnp.max(lengths))
    base = jnp.arange(max_len)[None, :]
    return base < lengths[:, None]


def get_position_ids(lengths: Array, max_length: Optional[int] = None) -> Array:
    max_len = max_length or int(jnp.max(lengths))
    base = jnp.arange(max_len)[None, :]
    return jnp.broadcast_to(base, (lengths.shape[0], max_len))


def rotate_half(x: Array) -> Array:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary(x: Array, cos: Array, sin: Array) -> Array:
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return (x * cos) + (rotate_half(x) * sin)


@dataclass
class EncoderOutput:
    hidden_states: Array
    packed_states: Array
    output_lengths: Array
    codes: Optional[Array]


@dataclass
class VocoderOutput:
    wav: Array
    wav_lengths: Array


@dataclass
class StreamingConfig:
    seg_point: int = field(default=60 * 25)
    process_seg_point: bool = field(default=True)
    left_overlap: int = field(default=10 * 25)
    right_overlap: int = field(default=40)
    seg_point_left_overlap: int = field(default=0)


@dataclass
class StreamingCache:
    hidden_states: Optional[List[Array]] = None
    processed_lengths: Optional[List[int]] = None


# though mimo-audio-tokenizer uses layer norm only, retained this method for compatibility.
def build_norm(name: str, dim: int, dtype, rngs: Optional[nnx.Rngs] = None) -> nnx.Module:
    if name == "RMSNorm":
        return nnx.RMSNorm(dim, epsilon=1e-6, param_dtype=dtype, rngs=rngs)
    return nnx.LayerNorm(dim, epsilon=1e-6, param_dtype=dtype, rngs=rngs)


class RotaryEmbedding(nnx.Module):
    def __init__(self, base: float, dim: int, max_seq_len: int, rope_type: str = "default", dtype=jnp.float32):
        self.base = base
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_type = rope_type
        self.dtype = dtype
        half_dim = dim // 2
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, half_dim, dtype=jnp.float32) / float(half_dim)))
        self.inv_freq = nnx.Param(inv_freq)
        self.attention_scaling = 1.0

    def __call__(self, hidden_states: Array, position_ids: Array) -> Tuple[Array, Array]:
        freq = position_ids[..., None] * self.inv_freq[None, None, :]
        emb = jnp.concatenate([freq, freq], axis=-1)
        cos = jnp.cos(emb) * self.attention_scaling
        sin = jnp.sin(emb) * self.attention_scaling
        return cos.astype(hidden_states.dtype), sin.astype(hidden_states.dtype)


class ConvTranspose1d(nnx.Module):
    """Custom 1D transposed convolution for specific audio processing requirements."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 shd_cfg: MiMoShardingCfg | None = None,
                 dtype=jnp.float32, rngs: Optional[nnx.Rngs] = None):
        self.stride = stride
        self.shd_cfg = shd_cfg or MiMoShardingCfg.no_sharding()

        # Create and shard kernel and bias
        kshape = (in_channels, out_channels, kernel_size)
        kernel = jnp.zeros(kshape, dtype=dtype)
        self.kernel = shard(nnx.Param(kernel), self.shd_cfg.conv_transpose_weight)

        bias = jnp.zeros((out_channels,), dtype=dtype)
        self.bias = shard(nnx.Param(bias), self.shd_cfg.conv_transpose_bias)

    def __call__(self, x: Array) -> Array:
        batch, length, channels = x.shape
        kernel = self.kernel.value
        kernel_size = kernel.shape[-1]
        up_len = (length - 1) * self.stride + 1
        idx = jnp.arange(length) * self.stride
        upsampled = jnp.zeros((batch, up_len, channels), dtype=x.dtype)
        upsampled = upsampled.at[:, idx, :].set(x)
        upsampled = jnp.pad(upsampled, ((0, 0), (kernel_size - 1, kernel_size - 1), (0, 0)))
        lhs = jnp.swapaxes(upsampled, 1, 2)
        rhs = jnp.flip(kernel, axis=-1).transpose(1, 0, 2)
        y = jax.lax.conv_general_dilated(
            lhs=lhs,
            rhs=rhs,
            window_strides=(1,),
            padding='VALID',
            dimension_numbers=('NCH', 'OIH', 'NCH'),
        )
        y = y + self.bias.value[None, :, None]
        y = jnp.swapaxes(y, 1, 2)
        return y


class ISTFT(nnx.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same",
                 shd_cfg: MiMoShardingCfg | None = None, dtype=jnp.float32):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.padding = padding
        self.shd_cfg = shd_cfg or MiMoShardingCfg.no_sharding()

        # Apply sharding to window parameter (usually replicated)
        self.window = shard(
            nnx.Param(jnp.hanning(win_length).astype(dtype)),
            self.shd_cfg.istft_window
        )

        self.pad = (self.win_length - self.hop_length) // 2 if padding == "same" else 0

    def __call__(self, spec: Array) -> Array:
        frames = jnp.fft.irfft(spec, n=self.n_fft, axis=1, norm="backward")
        frames = frames * self.window[None, :, None]
        frames = jnp.swapaxes(frames, 1, 2)
        batch, num_frames, _ = frames.shape
        output_size = (num_frames - 1) * self.hop_length + self.win_length
        audio = jnp.zeros((batch, output_size), dtype=frames.dtype)
        env = jnp.zeros_like(audio)
        window_sq = jnp.square(self.window)

        def body(i, carry):
            audio_acc, env_acc = carry
            start = i * self.hop_length
            frame = frames[:, i, :]
            current_audio = jax.lax.dynamic_slice(
                audio_acc,
                (0, start),
                (batch, self.win_length),
            )
            current_env = jax.lax.dynamic_slice(
                env_acc,
                (0, start),
                (batch, self.win_length),
            )
            updated_audio = current_audio + frame
            updated_env = current_env + window_sq
            audio_acc = jax.lax.dynamic_update_slice(audio_acc, updated_audio, (0, start))
            env_acc = jax.lax.dynamic_update_slice(env_acc, updated_env, (0, start))
            return audio_acc, env_acc

        audio, env = jax.lax.fori_loop(0, num_frames, body, (audio, env))
        # Trim padding before normalization (align with PyTorch implementation)
        if self.pad > 0:
            audio = audio[:, self.pad: -self.pad]
            env = env[:, self.pad: -self.pad]
        env = jnp.maximum(env, 1e-11)
        audio = audio / env
        return audio


class ISTFTHead(nnx.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same",
                 shd_cfg: MiMoShardingCfg | None = None,
                 dtype=jnp.float32, rngs: Optional[nnx.Rngs] = None):
        self.shd_cfg = shd_cfg or MiMoShardingCfg.no_sharding()

        # Apply sharding to Linear layer
        self.linear = shard(
            nnx.Linear(dim, n_fft + 2, dtype=dtype, rngs=rngs),
            self.shd_cfg.istft_linear_weight
        )

        # Pass shd_cfg to ISTFT
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
                          padding=padding, shd_cfg=self.shd_cfg, dtype=dtype)

    def __call__(self, hidden_states: Array) -> Array:
        x = self.linear(hidden_states)
        x = jnp.swapaxes(x, 1, 2)
        mag, phase = jnp.split(x, 2, axis=1)

        original_dtype = hidden_states.dtype
        mag = mag.astype(jnp.float32)
        phase = phase.astype(jnp.float32)

        mag = jnp.clip(jnp.exp(mag), a_max=1e2)
        real = jnp.cos(phase)
        imag = jnp.sin(phase)
        spec = mag * (real + 1j * imag)

        audio = self.istft(spec)
        audio = audio.astype(original_dtype)
        return audio


class Attention(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, window_size: Tuple[int, int], causal: bool,
                 shd_cfg: MiMoShardingCfg, dtype=jnp.float32,
                 rngs: Optional[nnx.Rngs] = None):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.window_size = window_size
        self.causal = causal
        self.shd_cfg = shd_cfg

        # Apply sharding to Linear layers
        self.q_proj = shard(
            nnx.Linear(embed_dim, embed_dim, use_bias=True, dtype=dtype, rngs=rngs),
            shd_cfg.attn_qkvo_weight
        )
        self.k_proj = shard(
            nnx.Linear(embed_dim, embed_dim, use_bias=False, dtype=dtype, rngs=rngs),
            shd_cfg.attn_qkvo_weight
        )
        self.v_proj = shard(
            nnx.Linear(embed_dim, embed_dim, use_bias=True, dtype=dtype, rngs=rngs),
            shd_cfg.attn_qkvo_weight
        )
        self.out_proj = shard(
            nnx.Linear(embed_dim, embed_dim, dtype=dtype, rngs=rngs),
            shd_cfg.attn_qkvo_weight
        )

    def _window_mask(self, seq_len: int) -> Optional[Array]:
        left, right = self.window_size
        if left < 0 and right < 0:
            return None
        pos = jnp.arange(seq_len)
        rel = pos[None, :] - pos[:, None]
        mask = jnp.ones((seq_len, seq_len), dtype=bool)
        if left >= 0:
            mask &= rel >= -left
        if right >= 0:
            mask &= rel <= right
        return mask

    def __call__(self, x: Array, mask: Optional[Array], rope: Optional[Tuple[Array, Array]]) -> Array:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        def reshape(t):
            t = t.reshape(batch, seq_len, self.num_heads, self.head_dim)
            return jnp.swapaxes(t, 1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Apply sharding to activations after reshape
        q = shard(q, self.shd_cfg.act_btnh)
        k = shard(k, self.shd_cfg.act_btnh)
        v = shard(v, self.shd_cfg.act_btnh)

        if rope is not None:
            cos, sin = rope
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        if mask is not None:
            scores = jnp.where(mask[:, None, None, :], scores, -1e9)
        if self.causal:
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
            scores = jnp.where(causal_mask, scores, -1e9)
        wmask = self._window_mask(seq_len)
        if wmask is not None:
            scores = jnp.where(wmask, scores, -1e9)
        weights = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        context = jnp.swapaxes(context, 1, 2).reshape(batch, seq_len, self.embed_dim)
        out = self.out_proj(context)
        if mask is not None:
            out = out * mask[..., None]

        # Apply sharding to output
        out = shard(out, self.shd_cfg.act_btd)
        return out


class TransformerLayer(nnx.Module):
    def __init__(self, d_model: int, attention_heads: int, ffn_dim: int, causal: bool,
                 attn_window_size: Tuple[int, int], shd_cfg: MiMoShardingCfg, dtype=jnp.float32,
                 rngs: Optional[nnx.Rngs] = None):
        self.act = jax.nn.gelu
        self.shd_cfg = shd_cfg

        # Pass shd_cfg to Attention
        self.self_attn = Attention(d_model, attention_heads, attn_window_size, causal,
                                   shd_cfg, dtype=dtype, rngs=rngs)

        # Apply sharding to LayerNorm layers
        self.self_attn_layer_norm = shard(
            nnx.LayerNorm(d_model, epsilon=1e-6, param_dtype=dtype, rngs=rngs),
            shd_cfg.norm_scale
        )
        self.final_layer_norm = shard(
            nnx.LayerNorm(d_model, epsilon=1e-6, param_dtype=dtype, rngs=rngs),
            shd_cfg.norm_scale
        )

        # Apply sharding to FFN layers
        self.fc1 = shard(
            nnx.Linear(d_model, ffn_dim, dtype=dtype, rngs=rngs),
            shd_cfg.ffn_weight_in
        )
        self.fc2 = shard(
            nnx.Linear(ffn_dim, d_model, dtype=dtype, rngs=rngs),
            shd_cfg.ffn_weight_out
        )

    def __call__(self, hidden_states: Array, mask: Optional[Array], rope: Optional[Tuple[Array, Array]]) -> Array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, mask, rope)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        # Apply sharding to activation after fc1
        hidden_states = shard(hidden_states, self.shd_cfg.act_btd)
        hidden_states = self.fc2(hidden_states)
        return residual + hidden_states


class ResidualVectorQuantizer(nnx.Module):
    def __init__(self, dimension: int, n_q: int, bins: Sequence[int],
                 shd_cfg: MiMoShardingCfg, dtype=jnp.float32,
                 rngs: Optional[nnx.Rngs] = None):
        self.dimension = dimension
        self.n_q = n_q
        self.shd_cfg = shd_cfg

        # Create codebooks with sharding applied
        codebooks_list = []
        for i in range(n_q):
            size = bins[min(i, len(bins) - 1)]
            embed = jnp.zeros((size, dimension), dtype=dtype)
            # Apply sharding to each codebook
            codebooks_list.append(shard(nnx.Param(embed), shd_cfg.codebook))
        self.codebooks = nnx.List(codebooks_list)

    def encode(self, hidden_states: Array, mask: Optional[Array] = None, n_q: Optional[int] = None) -> Tuple[
        Array, Array]:
        num_levels = n_q or self.n_q
        residual = hidden_states
        quantized = jnp.zeros_like(hidden_states)
        codes = []
        mask = None if mask is None else mask[..., None]
        for i in range(num_levels):
            codebook = self.codebooks[i].value
            dist = jnp.sum((residual[:, None, :] - codebook[None, :, :]) ** 2, axis=-1)
            idx = jnp.argmin(dist, axis=-1)
            chosen = codebook[idx]
            if mask is not None:
                chosen = chosen * mask
            quantized = quantized + chosen
            residual = residual - chosen
            codes.append(idx)
        return jnp.stack(codes, axis=0), quantized

    def decode(self, codes: Array) -> Array:
        num_levels = codes.shape[0]
        flat = codes.reshape(num_levels, -1)
        decoded = jnp.zeros((flat.shape[1], self.dimension), dtype=jnp.float32)
        for i in range(num_levels):
            codebook = self.codebooks[i].value
            decoded = decoded + codebook[flat[i]]
        return decoded.reshape(*codes.shape[1:], self.dimension)


class AudioEncoder(nnx.Module):
    def __init__(self, config: MiMoAudioTokenizerConfig, dtype=jnp.float32, rngs: Optional[nnx.Rngs] = None):
        self.config = config
        self.shd_cfg = config.shd_cfg

        # Apply sharding to Conv layers
        self.conv1 = shard(
            nnx.Conv(
                in_features=config.n_mels,
                out_features=config.d_model,
                kernel_size=config.kernel_size,
                padding="SAME",
                param_dtype=dtype,
                rngs=rngs
            ),
            self.shd_cfg.conv_weight
        )
        self.conv2 = shard(
            nnx.Conv(
                in_features=config.d_model,
                out_features=config.d_model,
                kernel_size=config.kernel_size,
                strides=config.stride_size,
                padding="SAME",
                param_dtype=dtype,
                rngs=rngs
            ),
            self.shd_cfg.conv_weight
        )

        # RotaryEmbedding doesn't need sharding (only inv_freq parameter, very small)
        self.position_embedding = RotaryEmbedding(config.rope_theta, config.d_model // config.encoder_attention_heads,
                                                  config.max_audio_seconds * config.sampling_rate // config.hop_length,
                                                  config.rope_type, dtype=dtype)

        # Pass shd_cfg to TransformerLayers
        self.layers = nnx.List([
            TransformerLayer(config.d_model, config.encoder_attention_heads, config.encoder_ffn_dim,
                             config.encoder_causal, tuple(config.encoder_attn_window_size),
                             self.shd_cfg, dtype=dtype, rngs=rngs)
            for _ in range(config.encoder_layers)
        ])

        # Apply sharding to LayerNorm
        self.layer_norm = shard(
            build_norm(config.ln_type, config.d_model, dtype, rngs),
            self.shd_cfg.norm_scale
        )

        if config.avg_pooler != 1:
            # Apply sharding to down-sample Conv layer
            self.down_sample_layer = shard(
                nnx.Conv(
                    in_features=config.d_model,
                    out_features=config.d_model,
                    kernel_size=config.avg_pooler,
                    strides=config.avg_pooler,
                    padding="SAME",
                    use_bias=False,
                    param_dtype=dtype,
                    rngs=rngs
                ),
                self.shd_cfg.conv_weight
            )
            self.down_norm = shard(
                build_norm(config.ln_type, config.d_model, dtype, rngs),
                self.shd_cfg.norm_scale
            )
        else:
            self.down_sample_layer = None
            self.down_norm = None

        if config.num_quantizers:
            bins = config.codebook_size or [1024]
            # Pass shd_cfg to ResidualVectorQuantizer
            self.quantizer = ResidualVectorQuantizer(config.d_model, config.num_quantizers, bins,
                                                     self.shd_cfg, dtype=dtype, rngs=rngs)
        else:
            self.quantizer = None

    def get_output_length(self, mel_len: Array) -> Array:
        tgt = mel_len + 3 - self.config.kernel_size
        return (tgt + 2 - self.config.kernel_size) // self.config.stride_size + 1

    def __call__(self, input_features: Array, input_lens: Array, use_quantizer: bool = True,
                 n_q: Optional[int] = None) -> EncoderOutput:
        x = input_features
        x = jax.nn.gelu(self.conv1(x))
        # Apply sharding to activation after conv1
        x = shard(x, self.shd_cfg.act_btd)

        x = jax.nn.gelu(self.conv2(x))
        # Apply sharding to activation after conv2
        x = shard(x, self.shd_cfg.act_btd)

        lengths = self.get_output_length(input_lens)
        max_len = x.shape[1]
        mask = make_sequence_mask(lengths, max_len)
        pos = get_position_ids(lengths, max_len)
        rope = self.position_embedding(x, pos)
        skip = 0.0
        for idx, layer in enumerate(self.layers):
            x = layer(x, mask, rope)
            if self.config.encoder_skip_layer_id and idx == self.config.encoder_skip_layer_id - 1:
                skip = x
        x = x + skip
        x = self.layer_norm(x)
        if self.down_sample_layer is not None:
            x = jax.nn.gelu(self.down_sample_layer(x))
            # Apply sharding to activation after down_sample
            x = shard(x, self.shd_cfg.act_btd)

            lengths = (lengths // self.config.avg_pooler) + ((lengths % self.config.avg_pooler) != 0).astype(
                lengths.dtype)
            max_len = x.shape[1]
            mask = make_sequence_mask(lengths, max_len)
            x = self.down_norm(x)
        x = x * mask[..., None]
        packed = x.reshape(-1, self.config.d_model)
        mask_flat = mask.reshape(-1)
        codes = None
        if self.quantizer is not None and use_quantizer:
            codes, quantized = self.quantizer.encode(packed, mask=mask_flat, n_q=n_q)
            packed = quantized
        packed = packed.reshape(x.shape)
        return EncoderOutput(hidden_states=packed, packed_states=packed, output_lengths=lengths, codes=codes)

    def decode_vq(self, codes: Array) -> Array:
        if self.quantizer is None:
            raise ValueError("Quantizer disabled")
        return self.quantizer.decode(codes)


class CausalConvTranspose1d(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 shd_cfg: MiMoShardingCfg | None = None,
                 dtype=jnp.float32, rngs: Optional[nnx.Rngs] = None):
        self.shd_cfg = shd_cfg or MiMoShardingCfg.no_sharding()

        # Pass shd_cfg to ConvTranspose1d
        self.conv = ConvTranspose1d(in_channels, out_channels, kernel_size, stride,
                                   shd_cfg=self.shd_cfg, dtype=dtype, rngs=rngs)

        # Apply sharding to GroupNorm
        self.norm = shard(
            nnx.GroupNorm(num_features=out_channels, num_groups=1, epsilon=1e-5,
                         param_dtype=dtype, rngs=rngs),
            self.shd_cfg.norm_scale
        )

        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x: Array, input_length: Array) -> Tuple[Array, Array]:
        y = self.conv(x)
        y = self.norm(y)
        trim = max(0, self.kernel_size - self.stride)
        if trim > 0:
            y = y[:, :-trim, :]
        output_len = (input_length - 1) * self.stride + self.kernel_size - trim
        return y, output_len


class TransformerVocos(nnx.Module):
    def __init__(self, config: MiMoAudioTokenizerConfig, dtype=jnp.float32, rngs: Optional[nnx.Rngs] = None):
        self.config = config
        self.shd_cfg = config.shd_cfg

        # Apply sharding to embeddings Linear layer
        self.embeddings = shard(
            nnx.Linear(config.n_mels, config.vocoder_dim, use_bias=False, dtype=dtype, rngs=rngs),
            self.shd_cfg.attn_qkvo_weight
        )

        # RotaryEmbedding doesn't need sharding (only inv_freq parameter, very small)
        self.position_embedding = RotaryEmbedding(config.rope_theta,
                                                  config.vocoder_dim // config.vocoder_attention_heads,
                                                  config.max_audio_seconds * config.sampling_rate // config.hop_length,
                                                  config.rope_type, dtype=dtype)

        # Pass shd_cfg to TransformerLayers
        self.layers = nnx.List([
            TransformerLayer(config.vocoder_dim, config.vocoder_attention_heads, config.vocoder_intermediate_dim,
                             False, tuple(config.vocoder_attn_window_size),
                             self.shd_cfg, dtype=dtype, rngs=rngs)
            for _ in range(config.vocoder_num_layers)
        ])

        # Apply sharding to LayerNorm
        self.layer_norm = shard(
            build_norm(config.ln_type, config.vocoder_dim, dtype, rngs),
            self.shd_cfg.norm_scale
        )

        # Pass shd_cfg to ISTFTHead
        self.head = ISTFTHead(config.vocoder_dim, config.nfft, config.hop_length,
                             config.vocoder_padding, shd_cfg=self.shd_cfg,
                             dtype=dtype, rngs=rngs)

    def __call__(self, mels: Array, input_length: Array) -> VocoderOutput:
        x = self.embeddings(mels)
        mask = make_sequence_mask(input_length, x.shape[1])
        pos = get_position_ids(input_length, x.shape[1])
        rope = self.position_embedding(x, pos)
        for layer in self.layers:
            x = layer(x, mask, rope)
        x = self.layer_norm(x)
        x = x * mask[..., None]
        wav = self.head(x)
        wav_len = input_length * self.config.hop_length
        wav = wav[:, None, :]
        return VocoderOutput(wav=wav, wav_lengths=wav_len)


class AudioDecoder(nnx.Module):
    def __init__(self, config: MiMoAudioTokenizerConfig, dtype=jnp.float32, rngs: Optional[nnx.Rngs] = None):
        self.config = config
        self.shd_cfg = config.shd_cfg

        if config.avg_pooler != 1:
            # Pass shd_cfg to CausalConvTranspose1d
            self.dconv1 = CausalConvTranspose1d(config.d_model, config.d_model, config.avg_pooler,
                                                config.avg_pooler, shd_cfg=self.shd_cfg,
                                                dtype=dtype, rngs=rngs)
        else:
            self.dconv1 = None

        # RotaryEmbedding doesn't need sharding (only inv_freq parameter, very small)
        self.position_embedding = RotaryEmbedding(config.rope_theta, config.d_model // config.decoder_attention_heads,
                                                  config.max_audio_seconds * config.sampling_rate // config.hop_length,
                                                  config.rope_type, dtype=dtype)

        # Pass shd_cfg to TransformerLayers
        self.layers = nnx.List([
            TransformerLayer(config.d_model, config.decoder_attention_heads, config.decoder_ffn_dim,
                             config.decoder_causal, tuple(config.decoder_attn_window_size),
                             self.shd_cfg, dtype=dtype, rngs=rngs)
            for _ in range(config.decoder_layers)
        ])

        # Apply sharding to LayerNorm
        self.layer_norm = shard(
            build_norm(config.ln_type, config.d_model, dtype, rngs),
            self.shd_cfg.norm_scale
        )

        # Pass shd_cfg to CausalConvTranspose1d
        self.dconv2 = CausalConvTranspose1d(config.d_model, config.n_mels, config.decoder_kernel_size,
                                            config.decoder_stride_size, shd_cfg=self.shd_cfg,
                                            dtype=dtype, rngs=rngs)

        # Pass config to TransformerVocos (it will get shd_cfg from config)
        self.vocoder = TransformerVocos(config, dtype=dtype, rngs=rngs)

    def __call__(self, audio_embed: Array, input_length: Array) -> Array:
        x = audio_embed
        lengths = input_length
        if self.dconv1 is not None:
            x, lengths = self.dconv1(x, lengths)
        mask = make_sequence_mask(lengths, x.shape[1])
        pos = get_position_ids(lengths, x.shape[1])
        rope = self.position_embedding(x, pos)
        for layer in self.layers:
            x = layer(x, mask, rope)
        x = self.layer_norm(x)
        coarse, mel_lengths = self.dconv2(x, lengths)
        vocoder_out = self.vocoder(coarse, mel_lengths)
        return vocoder_out.wav


class FlaxMiMoAudioTokenizer(nnx.Module):
    def __init__(self, config: MiMoAudioTokenizerConfig, dtype=jnp.float32, rngs: Optional[nnx.Rngs] = None):
        self.config = config
        self.encoder = AudioEncoder(config, dtype=dtype, rngs=rngs)
        self.decoder = AudioDecoder(config, dtype=dtype, rngs=rngs)
        self.downsample_rate = int(config.hop_length * 2 * config.avg_pooler)

    def __call__(self, mels: Array, input_lens: Array, use_quantizer: bool = True) -> Array:
        enc = self.encoder(mels, input_lens, use_quantizer=use_quantizer)
        return self.decoder(enc.hidden_states, enc.output_lengths)

    def encode(self, mels: Array, input_lens: Array, use_quantizer: bool = True,
               n_q: Optional[int] = None) -> EncoderOutput:
        return self.encoder(mels, input_lens, use_quantizer=use_quantizer, n_q=n_q)

    def decode(self, codes: Array) -> Array:
        hidden = self.encoder.decode_vq(codes)
        hidden = hidden[None, ...]  # 只添加batch dimension，不改变dtype
        lengths = jnp.array([hidden.shape[1]])
        return self.decoder(hidden, lengths)

    def streaming_decode(self, codes_chunks: Array, chunk_input_lengths: Sequence[int],
                         history_cache: Optional[StreamingCache] = None,
                         streaming_config: Optional[StreamingConfig] = None, last_chunk: bool = False):
        if history_cache is None:
            history_cache = StreamingCache()
        if streaming_config is None:
            streaming_config = StreamingConfig()
        hidden = self.encoder.decode_vq(codes_chunks)
        pointer = 0
        samples: List[Array] = []
        new_lengths: List[int] = []
        new_cache: List[Array] = []
        for idx, length in enumerate(chunk_input_lengths):
            sample = hidden[pointer: pointer + length]
            pointer += length
            if history_cache.hidden_states is not None:
                prev = history_cache.hidden_states[idx]
                sample = jnp.concatenate([prev, sample], axis=0)
                length = sample.shape[0]
            samples.append(sample)
            new_cache.append(sample)
            new_lengths.append(length)
        batch = len(samples)
        max_len = max(new_lengths)
        padded = []
        for sample in samples:
            pad = max_len - sample.shape[0]
            padded.append(jnp.pad(sample, ((0, pad), (0, 0))))
        decoder_in = jnp.stack(padded, axis=0)
        wavs = self.decoder(decoder_in, jnp.array(new_lengths))
        frames_per_token = self.config.avg_pooler * self.config.stride_size * self.config.hop_length
        processed: List[int] = []
        returned: List[Optional[Array]] = []
        for idx in range(batch):
            wav = wavs[idx]
            start = 0
            if history_cache.processed_lengths is not None:
                start = history_cache.processed_lengths[idx]
            if last_chunk:
                returned.append(wav[:, start * frames_per_token:])
                processed.append(new_lengths[idx])
                continue
            if new_lengths[idx] <= streaming_config.right_overlap:
                returned.append(None)
                processed.append(0)
                continue
            end = new_lengths[idx] - streaming_config.right_overlap
            clip = wav[:, start * frames_per_token: end * frames_per_token]
            returned.append(clip)
            new_processed = end
            if new_lengths[idx] > streaming_config.left_overlap:
                keep = streaming_config.left_overlap
                new_cache[idx] = new_cache[idx][-keep:]
                new_processed -= new_lengths[idx] - keep
            processed.append(new_processed)
        history_cache.hidden_states = new_cache
        history_cache.processed_lengths = processed
        return returned, history_cache
