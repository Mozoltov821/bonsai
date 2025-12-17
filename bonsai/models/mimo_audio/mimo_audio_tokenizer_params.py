"""Weight loading for MiMo Audio Tokenizer from SafeTensors"""

import gc
import re
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from safetensors import safe_open

from bonsai.models.mimo_audio import mimo_audio_tokenizer as model_lib


@dataclass(frozen=True)
class Transform:
    permute: tuple[int, ...] | None = None
    reshape: tuple[int, ...] | None = None
    reshape_first: bool = False


TRANSFORM_LINEAR = Transform(permute=(1, 0))
TRANSFORM_CONV1D = Transform(permute=(2, 1, 0))
TRANSFORM_NONE = Transform()


def _get_key_mapping(config: model_lib.MiMoAudioTokenizerConfig) -> dict[str, tuple[str, Transform]]:
    mapping = {
        r"encoder\.conv1\.weight": ("encoder.conv1.kernel", TRANSFORM_CONV1D),
        r"encoder\.conv1\.bias": ("encoder.conv1.bias", TRANSFORM_NONE),
        r"encoder\.conv2\.weight": ("encoder.conv2.kernel", TRANSFORM_CONV1D),
        r"encoder\.conv2\.bias": ("encoder.conv2.bias", TRANSFORM_NONE),
        r"encoder\.layer_norm\.weight": ("encoder.layer_norm.scale", TRANSFORM_NONE),
        r"encoder\.layer_norm\.bias": ("encoder.layer_norm.bias", TRANSFORM_NONE),
    }

    for idx in range(config.encoder_layers):
        layer_mappings = {
            rf"encoder\.layers\.{idx}\.self_attn\.q_proj\.weight": (f"encoder.layers.{idx}.self_attn.q_proj.kernel", TRANSFORM_LINEAR),
            rf"encoder\.layers\.{idx}\.self_attn\.q_proj\.bias": (f"encoder.layers.{idx}.self_attn.q_proj.bias", TRANSFORM_NONE),
            rf"encoder\.layers\.{idx}\.self_attn\.k_proj\.weight": (f"encoder.layers.{idx}.self_attn.k_proj.kernel", TRANSFORM_LINEAR),
            rf"encoder\.layers\.{idx}\.self_attn\.k_proj\.bias": (f"encoder.layers.{idx}.self_attn.k_proj.bias", TRANSFORM_NONE),
            rf"encoder\.layers\.{idx}\.self_attn\.v_proj\.weight": (f"encoder.layers.{idx}.self_attn.v_proj.kernel", TRANSFORM_LINEAR),
            rf"encoder\.layers\.{idx}\.self_attn\.v_proj\.bias": (f"encoder.layers.{idx}.self_attn.v_proj.bias", TRANSFORM_NONE),
            rf"encoder\.layers\.{idx}\.self_attn\.out_proj\.weight": (f"encoder.layers.{idx}.self_attn.out_proj.kernel", TRANSFORM_LINEAR),
            rf"encoder\.layers\.{idx}\.self_attn\.out_proj\.bias": (f"encoder.layers.{idx}.self_attn.out_proj.bias", TRANSFORM_NONE),
            rf"encoder\.layers\.{idx}\.self_attn_layer_norm\.weight": (f"encoder.layers.{idx}.self_attn_layer_norm.scale", TRANSFORM_NONE),
            rf"encoder\.layers\.{idx}\.self_attn_layer_norm\.bias": (f"encoder.layers.{idx}.self_attn_layer_norm.bias", TRANSFORM_NONE),
            rf"encoder\.layers\.{idx}\.final_layer_norm\.weight": (f"encoder.layers.{idx}.final_layer_norm.scale", TRANSFORM_NONE),
            rf"encoder\.layers\.{idx}\.final_layer_norm\.bias": (f"encoder.layers.{idx}.final_layer_norm.bias", TRANSFORM_NONE),
            rf"encoder\.layers\.{idx}\.fc1\.weight": (f"encoder.layers.{idx}.fc1.kernel", TRANSFORM_LINEAR),
            rf"encoder\.layers\.{idx}\.fc1\.bias": (f"encoder.layers.{idx}.fc1.bias", TRANSFORM_NONE),
            rf"encoder\.layers\.{idx}\.fc2\.weight": (f"encoder.layers.{idx}.fc2.kernel", TRANSFORM_LINEAR),
            rf"encoder\.layers\.{idx}\.fc2\.bias": (f"encoder.layers.{idx}.fc2.bias", TRANSFORM_NONE),
        }
        mapping.update(layer_mappings)

    mapping.update({
        r"encoder\.down_sample_layer\.0\.weight": ("encoder.down_sample_layer.kernel", TRANSFORM_CONV1D),
        r"encoder\.down_sample_layer\.0\.bias": ("encoder.down_sample_layer.bias", TRANSFORM_NONE),
        r"encoder\.down_sample_norm\.weight": ("encoder.down_norm.scale", TRANSFORM_NONE),
        r"encoder\.down_sample_norm\.bias": ("encoder.down_norm.bias", TRANSFORM_NONE),
    })

    for idx in range(config.num_quantizers):
        mapping[rf"encoder\.quantizer\.vq\.layers\.{idx}\._codebook\.embed"] = (
            f"encoder.quantizer.codebooks.{idx}",
            TRANSFORM_NONE,
        )

    mapping.update({
        r"decoder\.dconv1\.conv\.weight": ("decoder.dconv1.conv.kernel", TRANSFORM_NONE),
        r"decoder\.dconv1\.conv\.bias": ("decoder.dconv1.conv.bias", TRANSFORM_NONE),
        r"decoder\.dconv1\.norm\.weight": ("decoder.dconv1.norm.scale", TRANSFORM_NONE),
        r"decoder\.dconv1\.norm\.bias": ("decoder.dconv1.norm.bias", TRANSFORM_NONE),
        r"decoder\.layer_norm\.weight": ("decoder.layer_norm.scale", TRANSFORM_NONE),
        r"decoder\.layer_norm\.bias": ("decoder.layer_norm.bias", TRANSFORM_NONE),
        r"decoder\.dconv2\.conv\.weight": ("decoder.dconv2.conv.kernel", TRANSFORM_NONE),
        r"decoder\.dconv2\.conv\.bias": ("decoder.dconv2.conv.bias", TRANSFORM_NONE),
        r"decoder\.dconv2\.norm\.weight": ("decoder.dconv2.norm.scale", TRANSFORM_NONE),
        r"decoder\.dconv2\.norm\.bias": ("decoder.dconv2.norm.bias", TRANSFORM_NONE),
    })

    for idx in range(config.decoder_layers):
        layer_mappings = {
            rf"decoder\.layers\.{idx}\.self_attn\.q_proj\.weight": (f"decoder.layers.{idx}.self_attn.q_proj.kernel", TRANSFORM_LINEAR),
            rf"decoder\.layers\.{idx}\.self_attn\.q_proj\.bias": (f"decoder.layers.{idx}.self_attn.q_proj.bias", TRANSFORM_NONE),
            rf"decoder\.layers\.{idx}\.self_attn\.k_proj\.weight": (f"decoder.layers.{idx}.self_attn.k_proj.kernel", TRANSFORM_LINEAR),
            rf"decoder\.layers\.{idx}\.self_attn\.k_proj\.bias": (f"decoder.layers.{idx}.self_attn.k_proj.bias", TRANSFORM_NONE),
            rf"decoder\.layers\.{idx}\.self_attn\.v_proj\.weight": (f"decoder.layers.{idx}.self_attn.v_proj.kernel", TRANSFORM_LINEAR),
            rf"decoder\.layers\.{idx}\.self_attn\.v_proj\.bias": (f"decoder.layers.{idx}.self_attn.v_proj.bias", TRANSFORM_NONE),
            rf"decoder\.layers\.{idx}\.self_attn\.out_proj\.weight": (f"decoder.layers.{idx}.self_attn.out_proj.kernel", TRANSFORM_LINEAR),
            rf"decoder\.layers\.{idx}\.self_attn\.out_proj\.bias": (f"decoder.layers.{idx}.self_attn.out_proj.bias", TRANSFORM_NONE),
            rf"decoder\.layers\.{idx}\.self_attn_layer_norm\.weight": (f"decoder.layers.{idx}.self_attn_layer_norm.scale", TRANSFORM_NONE),
            rf"decoder\.layers\.{idx}\.self_attn_layer_norm\.bias": (f"decoder.layers.{idx}.self_attn_layer_norm.bias", TRANSFORM_NONE),
            rf"decoder\.layers\.{idx}\.final_layer_norm\.weight": (f"decoder.layers.{idx}.final_layer_norm.scale", TRANSFORM_NONE),
            rf"decoder\.layers\.{idx}\.final_layer_norm\.bias": (f"decoder.layers.{idx}.final_layer_norm.bias", TRANSFORM_NONE),
            rf"decoder\.layers\.{idx}\.fc1\.weight": (f"decoder.layers.{idx}.fc1.kernel", TRANSFORM_LINEAR),
            rf"decoder\.layers\.{idx}\.fc1\.bias": (f"decoder.layers.{idx}.fc1.bias", TRANSFORM_NONE),
            rf"decoder\.layers\.{idx}\.fc2\.weight": (f"decoder.layers.{idx}.fc2.kernel", TRANSFORM_LINEAR),
            rf"decoder\.layers\.{idx}\.fc2\.bias": (f"decoder.layers.{idx}.fc2.bias", TRANSFORM_NONE),
        }
        mapping.update(layer_mappings)

    mapping.update({
        r"decoder\.vocoder\.embeddings\.weight": ("decoder.vocoder.embeddings.kernel", TRANSFORM_LINEAR),
        r"decoder\.vocoder\.embeddings\.bias": ("decoder.vocoder.embeddings.bias", TRANSFORM_NONE),
        r"decoder\.vocoder\.layer_norm\.weight": ("decoder.vocoder.layer_norm.scale", TRANSFORM_NONE),
        r"decoder\.vocoder\.layer_norm\.bias": ("decoder.vocoder.layer_norm.bias", TRANSFORM_NONE),
        r"decoder\.vocoder\.head\.out\.weight": ("decoder.vocoder.head.linear.kernel", TRANSFORM_LINEAR),
        r"decoder\.vocoder\.head\.out\.bias": ("decoder.vocoder.head.linear.bias", TRANSFORM_NONE),
        r"decoder\.vocoder\.head\.istft\.window": ("decoder.vocoder.head.istft.window", TRANSFORM_NONE),
    })

    for idx in range(config.vocoder_num_layers):
        layer_mappings = {
            rf"decoder\.vocoder\.layers\.{idx}\.self_attn\.q_proj\.weight": (f"decoder.vocoder.layers.{idx}.self_attn.q_proj.kernel", TRANSFORM_LINEAR),
            rf"decoder\.vocoder\.layers\.{idx}\.self_attn\.q_proj\.bias": (f"decoder.vocoder.layers.{idx}.self_attn.q_proj.bias", TRANSFORM_NONE),
            rf"decoder\.vocoder\.layers\.{idx}\.self_attn\.k_proj\.weight": (f"decoder.vocoder.layers.{idx}.self_attn.k_proj.kernel", TRANSFORM_LINEAR),
            rf"decoder\.vocoder\.layers\.{idx}\.self_attn\.k_proj\.bias": (f"decoder.vocoder.layers.{idx}.self_attn.k_proj.bias", TRANSFORM_NONE),
            rf"decoder\.vocoder\.layers\.{idx}\.self_attn\.v_proj\.weight": (f"decoder.vocoder.layers.{idx}.self_attn.v_proj.kernel", TRANSFORM_LINEAR),
            rf"decoder\.vocoder\.layers\.{idx}\.self_attn\.v_proj\.bias": (f"decoder.vocoder.layers.{idx}.self_attn.v_proj.bias", TRANSFORM_NONE),
            rf"decoder\.vocoder\.layers\.{idx}\.self_attn\.out_proj\.weight": (f"decoder.vocoder.layers.{idx}.self_attn.out_proj.kernel", TRANSFORM_LINEAR),
            rf"decoder\.vocoder\.layers\.{idx}\.self_attn\.out_proj\.bias": (f"decoder.vocoder.layers.{idx}.self_attn.out_proj.bias", TRANSFORM_NONE),
            rf"decoder\.vocoder\.layers\.{idx}\.self_attn_layer_norm\.weight": (f"decoder.vocoder.layers.{idx}.self_attn_layer_norm.scale", TRANSFORM_NONE),
            rf"decoder\.vocoder\.layers\.{idx}\.self_attn_layer_norm\.bias": (f"decoder.vocoder.layers.{idx}.self_attn_layer_norm.bias", TRANSFORM_NONE),
            rf"decoder\.vocoder\.layers\.{idx}\.final_layer_norm\.weight": (f"decoder.vocoder.layers.{idx}.final_layer_norm.scale", TRANSFORM_NONE),
            rf"decoder\.vocoder\.layers\.{idx}\.final_layer_norm\.bias": (f"decoder.vocoder.layers.{idx}.final_layer_norm.bias", TRANSFORM_NONE),
            rf"decoder\.vocoder\.layers\.{idx}\.fc1\.weight": (f"decoder.vocoder.layers.{idx}.fc1.kernel", TRANSFORM_LINEAR),
            rf"decoder\.vocoder\.layers\.{idx}\.fc1\.bias": (f"decoder.vocoder.layers.{idx}.fc1.bias", TRANSFORM_NONE),
            rf"decoder\.vocoder\.layers\.{idx}\.fc2\.weight": (f"decoder.vocoder.layers.{idx}.fc2.kernel", TRANSFORM_LINEAR),
            rf"decoder\.vocoder\.layers\.{idx}\.fc2\.bias": (f"decoder.vocoder.layers.{idx}.fc2.bias", TRANSFORM_NONE),
        }
        mapping.update(layer_mappings)

    return mapping


def _get_jax_key(
    mapping: dict[str, tuple[str, Transform]], source_key: str
) -> tuple[str | None, Transform | None]:
    for pat, (jax_key, transform) in mapping.items():
        if re.fullmatch(pat, source_key):
            return jax_key, transform
    return None, None


def _assign_weights(
    keys: list[str | int],
    tensor: Any,
    state_dict: dict,
    transform: Transform | None,
) -> None:
    key, *rest = keys
    if not rest:
        if transform is not None:
            if transform.reshape_first and transform.reshape is not None:
                tensor = tensor.reshape(transform.reshape)
            if transform.permute is not None:
                tensor = tensor.transpose(transform.permute)
            if not transform.reshape_first and transform.reshape is not None:
                tensor = tensor.reshape(transform.reshape)

        if tensor.shape != state_dict[key].shape:
            raise ValueError(f"Shape mismatch: {tensor.shape} vs {state_dict[key].shape}")

        state_dict[key] = jax.device_put(tensor)
    else:
        _assign_weights(rest, tensor, state_dict[key], transform)


def _stoi(s: str) -> str | int:
    try:
        return int(s)
    except ValueError:
        return s


def load_tokenizer_weights_from_safetensors(
    config: model_lib.MiMoAudioTokenizerConfig,
    safetensors_path: str,
    dtype=jnp.float32,
    rngs: nnx.Rngs | None = None,
) -> model_lib.FlaxMiMoAudioTokenizer:
    if rngs is None:
        rngs = nnx.Rngs(params=0)

    model = nnx.eval_shape(lambda: model_lib.FlaxMiMoAudioTokenizer(config, dtype=dtype, rngs=rngs))
    graph_def, abs_state = nnx.split(model)
    state_dict = abs_state.to_pure_dict()

    key_mapping = _get_key_mapping(config)
    conversion_errors = []
    skipped_keys = []

    with safe_open(safetensors_path, framework="numpy") as sf:
        for torch_key in sf.keys():
            jax_key, transform = _get_jax_key(key_mapping, torch_key)
            if jax_key is None:
                skipped_keys.append(torch_key)
                continue

            keys = [_stoi(k) for k in jax_key.split(".")]
            try:
                tensor = sf.get_tensor(torch_key)
                _assign_weights(keys, tensor, state_dict, transform)
            except Exception as e:
                conversion_errors.append(f"Failed to assign '{torch_key}' to '{jax_key}': {type(e).__name__}: {e}")

    if skipped_keys:
        print(f"Warning: Skipped {len(skipped_keys)} keys not found in mapping")

    if conversion_errors:
        raise RuntimeError(
            f"Encountered {len(conversion_errors)} weight conversion errors:\n" + "\n".join(conversion_errors)
        )

    model = nnx.merge(graph_def, state_dict)
    gc.collect()

    print("Tokenizer weights loaded successfully!")
    return model
