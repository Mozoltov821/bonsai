import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple
import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

Array = jnp.ndarray


class MiMoAudioTokenizerConfig(PretrainedConfig):
    model_type = "mimo_audio_tokenizer"

    def __init__(
            self,
            max_audio_seconds: int = 1800,
            stride_size: int = 2,
            avg_pooler: int = 1,
            d_model: int = 768,
            scale_embedding: bool = True,
            kernel_size: int = 3,
            activation_function: str = "gelu",
            encoder_layers: int = 8,
            encoder_skip_layer_id: int = None,
            encoder_attention_heads: int = 12,
            encoder_ffn_dim: int = 3072,
            encoder_causal: bool = False,
            encoder_attn_window_size: list[int] = None,
            decoder_layers: int = 8,
            decoder_attention_heads: int = 12,
            decoder_ffn_dim: int = 3072,
            decoder_kernel_size: int = 3,
            decoder_stride_size: int = 2,
            decoder_causal: bool = True,
            decoder_attn_window_size: list[int] = None,
            nfft: int = 1024,
            vocoder_dim: int = 512,
            vocoder_intermediate_dim: int = 4096,
            vocoder_num_layers: int = 30,
            n_mels: int = 80,
            sampling_rate: int = 24000,
            hop_length: int = 240,
            window_size: int = 1024,
            vocoder_padding: str = "same",
            fmin: int = 0,
            fmax: int = None,
            num_quantizers: int = 12,
            codebook_size: list[int] = None,
            threshold_ema_dead_code: int = 10,
            position_embedding_type: str = "rope",
            rope_theta: int = 10000,
            rope_type: str = "default",
            ln_type: str = "LayerNorm",
            vocoder_attention_heads: int = 4,
            vocoder_attn_window_size: list[int] = None,
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


def _default_activation(name: str) -> Callable[[Array], Array]:
    name = name.lower()
    if name == "relu":
        return jax.nn.relu
    if name in ("gelu", "gelu_new"):
        return jax.nn.gelu
    if name in ("silu", "swish"):
        return jax.nn.silu
    if name == "tanh":
        return jnp.tanh
    raise ValueError(f"Unsupported activation {name}")


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


def Conv1d(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
           padding: str = "SAME", use_bias: bool = True, dtype=jnp.float32, rngs: Optional[nnx.Rngs] = None):
    """Wrapper around nnx.Conv for 1D convolution with zero initialization."""
    return nnx.Conv(
        in_features=in_channels,
        out_features=out_channels,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        use_bias=use_bias,
        param_dtype=dtype,
        kernel_init=nnx.initializers.zeros_init(),
        bias_init=nnx.initializers.zeros_init(),
        rngs=rngs
    )


class ConvTranspose1d(nnx.Module):
    """Custom 1D transposed convolution for specific audio processing requirements."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 dtype=jnp.float32, rngs: Optional[nnx.Rngs] = None):
        self.stride = stride
        kshape = (in_channels, out_channels, kernel_size)
        kernel = jnp.zeros(kshape, dtype=dtype)
        self.kernel = nnx.Param(kernel)
        self.bias = nnx.Param(jnp.zeros((out_channels,), dtype=dtype))

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
    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same", dtype=jnp.float32):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.padding = padding
        self.window = nnx.Param(jnp.hanning(win_length).astype(dtype))
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
        env = jnp.maximum(env, 1e-11)
        audio = audio / env
        if self.pad > 0:
            audio = audio[:, self.pad: -self.pad]
        return audio


class ISTFTHead(nnx.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same", dtype=jnp.float32,
                 rngs: Optional[nnx.Rngs] = None):
        self.linear = nnx.Linear(dim, n_fft + 2, dtype=dtype, rngs=rngs)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding, dtype=dtype)

    def __call__(self, hidden_states: Array) -> Array:
        x = self.linear(hidden_states)
        x = jnp.swapaxes(x, 1, 2)
        mag, phase = jnp.split(x, 2, axis=1)
        mag = jnp.clip(jnp.exp(mag), a_max=1e2)
        real = jnp.cos(phase)
        imag = jnp.sin(phase)
        spec = mag * (real + 1j * imag)
        return self.istft(spec)


class Attention(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, window_size: Tuple[int, int], causal: bool, dtype=jnp.float32,
                 rngs: Optional[nnx.Rngs] = None):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.window_size = window_size
        self.causal = causal
        self.q_proj = nnx.Linear(embed_dim, embed_dim, use_bias=True, dtype=dtype, rngs=rngs)
        self.k_proj = nnx.Linear(embed_dim, embed_dim, use_bias=False, dtype=dtype, rngs=rngs)
        self.v_proj = nnx.Linear(embed_dim, embed_dim, use_bias=True, dtype=dtype, rngs=rngs)
        self.out_proj = nnx.Linear(embed_dim, embed_dim, dtype=dtype, rngs=rngs)

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
        return out


class TransformerLayer(nnx.Module):
    def __init__(self, act: Callable[[Array], Array], d_model: int, attention_heads: int, ffn_dim: int, causal: bool,
                 ln_type: str, attn_window_size: Tuple[int, int], dtype=jnp.float32, rngs: Optional[nnx.Rngs] = None):
        self.act = act
        self.self_attn = Attention(d_model, attention_heads, attn_window_size, causal, dtype=dtype, rngs=rngs)
        self.self_attn_layer_norm = build_norm(ln_type, d_model, dtype, rngs)
        self.final_layer_norm = build_norm(ln_type, d_model, dtype, rngs)
        self.fc1 = nnx.Linear(d_model, ffn_dim, dtype=dtype, rngs=rngs)
        self.fc2 = nnx.Linear(ffn_dim, d_model, dtype=dtype, rngs=rngs)

    def __call__(self, x: Array, mask: Optional[Array], rope: Optional[Tuple[Array, Array]]) -> Array:
        residual = x
        y = self.self_attn_layer_norm(x)
        y = self.self_attn(y, mask, rope)
        x = residual + y
        residual = x
        y = self.final_layer_norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        return residual + y


class ResidualVectorQuantizer(nnx.Module):
    def __init__(self, dimension: int, n_q: int, bins: Sequence[int], dtype=jnp.float32,
                 rngs: Optional[nnx.Rngs] = None):
        self.dimension = dimension
        self.n_q = n_q
        codebooks_list = []
        for i in range(n_q):
            size = bins[min(i, len(bins) - 1)]
            embed = jnp.zeros((size, dimension), dtype=dtype)
            codebooks_list.append(nnx.Param(embed))
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
        self.conv1 = Conv1d(config.n_mels, config.d_model, kernel_size=config.kernel_size, padding="SAME", dtype=dtype,
                            rngs=rngs)
        self.conv2 = Conv1d(config.d_model, config.d_model, kernel_size=config.kernel_size, stride=config.stride_size,
                            padding="SAME", dtype=dtype, rngs=rngs)
        self.position_embedding = RotaryEmbedding(config.rope_theta, config.d_model // config.encoder_attention_heads,
                                                  config.max_audio_seconds * config.sampling_rate // config.hop_length,
                                                  config.rope_type, dtype=dtype)
        act = _default_activation(config.activation_function)
        self.layers = nnx.List([
            TransformerLayer(act, config.d_model, config.encoder_attention_heads, config.encoder_ffn_dim,
                             config.encoder_causal, config.ln_type, tuple(config.encoder_attn_window_size), dtype=dtype,
                             rngs=rngs)
            for _ in range(config.encoder_layers)
        ])
        self.layer_norm = build_norm(config.ln_type, config.d_model, dtype, rngs)
        if config.avg_pooler != 1:
            self.down_sample_layer = Conv1d(config.d_model, config.d_model, kernel_size=config.avg_pooler,
                                            stride=config.avg_pooler, padding="SAME", use_bias=False, dtype=dtype,
                                            rngs=rngs)
            self.down_norm = build_norm(config.ln_type, config.d_model, dtype, rngs)
        else:
            self.down_sample_layer = None
            self.down_norm = None
        if config.num_quantizers:
            bins = config.codebook_size or [1024]
            self.quantizer = ResidualVectorQuantizer(config.d_model, config.num_quantizers, bins, dtype=dtype,
                                                     rngs=rngs)
        else:
            self.quantizer = None

    def get_output_length(self, mel_len: Array) -> Array:
        tgt = mel_len + 3 - self.config.kernel_size
        return (tgt + 2 - self.config.kernel_size) // self.config.stride_size + 1

    def __call__(self, input_features: Array, input_lens: Array, use_quantizer: bool = True,
                 n_q: Optional[int] = None) -> EncoderOutput:
        x = input_features
        x = jax.nn.gelu(self.conv1(x))
        x = jax.nn.gelu(self.conv2(x))
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dtype=jnp.float32,
                 rngs: Optional[nnx.Rngs] = None):
        self.conv = ConvTranspose1d(in_channels, out_channels, kernel_size, stride, dtype=dtype, rngs=rngs)
        self.norm = nnx.GroupNorm(num_features=out_channels, num_groups=1, epsilon=1e-5, param_dtype=dtype, rngs=rngs)
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
        self.embeddings = nnx.Linear(config.n_mels, config.vocoder_dim, use_bias=False, dtype=dtype, rngs=rngs)
        self.position_embedding = RotaryEmbedding(config.rope_theta,
                                                  config.vocoder_dim // config.vocoder_attention_heads,
                                                  config.max_audio_seconds * config.sampling_rate // config.hop_length,
                                                  config.rope_type, dtype=dtype)
        act = _default_activation(config.activation_function)
        self.layers = nnx.List([
            TransformerLayer(act, config.vocoder_dim, config.vocoder_attention_heads, config.vocoder_intermediate_dim,
                             False, config.ln_type, tuple(config.vocoder_attn_window_size), dtype=dtype, rngs=rngs)
            for _ in range(config.vocoder_num_layers)
        ])
        self.layer_norm = build_norm(config.ln_type, config.vocoder_dim, dtype, rngs)
        self.head = ISTFTHead(config.vocoder_dim, config.nfft, config.hop_length, config.vocoder_padding, dtype=dtype,
                              rngs=rngs)

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
        if config.avg_pooler != 1:
            self.dconv1 = CausalConvTranspose1d(config.d_model, config.d_model, config.avg_pooler, config.avg_pooler,
                                                dtype=dtype, rngs=rngs)
        else:
            self.dconv1 = None
        self.position_embedding = RotaryEmbedding(config.rope_theta, config.d_model // config.decoder_attention_heads,
                                                  config.max_audio_seconds * config.sampling_rate // config.hop_length,
                                                  config.rope_type, dtype=dtype)
        act = _default_activation(config.activation_function)
        self.layers = nnx.List([
            TransformerLayer(act, config.d_model, config.decoder_attention_heads, config.decoder_ffn_dim,
                             config.decoder_causal, config.ln_type, tuple(config.decoder_attn_window_size), dtype=dtype,
                             rngs=rngs)
            for _ in range(config.decoder_layers)
        ])
        self.layer_norm = build_norm(config.ln_type, config.d_model, dtype, rngs)
        self.dconv2 = CausalConvTranspose1d(config.d_model, config.n_mels, config.decoder_kernel_size,
                                            config.decoder_stride_size, dtype=dtype, rngs=rngs)
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
        # ✅ 修复：保持float32精度，不转换为bfloat16
        # PyTorch实现也保持decode_vq的float32输出，确保音频质量
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
