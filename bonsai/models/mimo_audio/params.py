"""MiMo Audio Weight Loading"""

from enum import Enum
import jax.numpy as jnp
from flax import nnx
from bonsai.models.qwen2 import params as qwen2_params


def _load_tensor(state_dict: dict, key: str, transpose: bool = False) -> jnp.ndarray:
    tensor = state_dict[key]
    if transpose:
        tensor = tensor.T
    return jnp.array(tensor, dtype=jnp.bfloat16)


def load_qwen2_weights(qwen2_model, state_dict: dict, prefix: str, cfg=None):

    class Transform(Enum):
        BIAS = None
        LINEAR = ((1, 0), None, False)
        EMBED = None
        SCALE = None
        SKIP = None

    key_mapping = {
        rf"{prefix}\.embed_tokens\.weight": ("embedder.embedding", Transform.EMBED),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (r"layers.\1.attn.q_proj.kernel", Transform.LINEAR),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (r"layers.\1.attn.k_proj.kernel", Transform.LINEAR),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (r"layers.\1.attn.v_proj.kernel", Transform.LINEAR),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (r"layers.\1.attn.o_proj.kernel", Transform.LINEAR),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": (r"layers.\1.attn.q_bias", Transform.BIAS),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": (r"layers.\1.attn.k_bias", Transform.BIAS),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": (r"layers.\1.attn.v_bias", Transform.BIAS),
        rf"{prefix}\.layers\.([0-9]+)\.self_attn\.o_proj\.bias": (r"layers.\1.attn.o_bias", Transform.BIAS),
        rf"{prefix}\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (r"layers.\1.mlp.gate_proj.kernel", Transform.LINEAR),
        rf"{prefix}\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (r"layers.\1.mlp.up_proj.kernel", Transform.LINEAR),
        rf"{prefix}\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (r"layers.\1.mlp.down_proj.kernel", Transform.LINEAR),
        rf"{prefix}\.norm\.weight": ("final_norm.scale", Transform.SCALE),
        rf"{prefix}\.layers\.([0-9]+)\.input_layernorm\.weight": (r"layers.\1.input_layernorm.scale", Transform.SCALE),
        rf"{prefix}\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
        r"layers.\1.post_attention_layernorm.scale", Transform.SCALE),
    }

    loaded_count = 0
    bias_count = 0
    for torch_key in state_dict.keys():
        if not torch_key.startswith(prefix):
            continue

        jax_key, transform = qwen2_params._torch_key_to_jax_key(key_mapping, torch_key)
        if jax_key is None:
            continue

        tensor = state_dict[torch_key]
        if transform and transform.value:
            permute, reshape, reshape_first = transform.value
            if permute:
                tensor = tensor.transpose(permute)

        tensor = jnp.array(tensor, dtype=jnp.bfloat16)

        keys = [qwen2_params._stoi(k) for k in jax_key.split(".")]
        obj = qwen2_model
        for key in keys[:-1]:
            obj = getattr(obj, key) if isinstance(key, str) else obj[key]

        final_key = keys[-1]

        if transform == Transform.BIAS:
            setattr(obj, final_key, nnx.Param(tensor))
            bias_count += 1
        else:
            target = getattr(obj, final_key) if isinstance(final_key, str) else obj[final_key]

            if hasattr(target, 'value'):
                target.value = tensor
            else:
                if isinstance(final_key, str):
                    setattr(obj, final_key, tensor)
                else:
                    obj[final_key] = tensor

            loaded_count += 1

    return loaded_count


def load_mimo_audio_weights(model, state_dict: dict):
    """Load all weights for MiMo Audio model"""
    print("Loading MiMo Audio weights from safetensors...")

    load_qwen2_weights(model.model, state_dict, "model", model.qwen2_config)
    load_qwen2_weights(model.local_transformer, state_dict, "local_transformer", model.local_qwen2_config)
    load_qwen2_weights(model.input_local_transformer, state_dict, "input_local_transformer",
                       model.input_local_qwen2_config)

    if "lm_head.weight" in state_dict:
        model.model.lm_head.w.value = _load_tensor(state_dict, "lm_head.weight", transpose=True)
        print("Loaded lm_head")

    for i in range(model.audio_channels):
        key = f"speech_embeddings.{i}.weight"
        if key in state_dict:
            model.speech_embeddings[i].embedding.value = _load_tensor(state_dict, key)
    print(f"Loaded {model.audio_channels} speech embeddings")

    for i in range(model.audio_channels):
        key = f"local_transformer_lm_heads.{i}.weight"
        if key in state_dict:
            model.local_transformer_lm_heads[i].kernel.value = _load_tensor(state_dict, key, transpose=True)
    print(f"Loaded {model.audio_channels} local LM heads")

    if "hidden_states_downcast.weight" in state_dict:
        model.hidden_states_downcast.kernel.value = _load_tensor(state_dict, "hidden_states_downcast.weight",
                                                                 transpose=True)
        print("Loaded hidden_states_downcast")

    if "speech_group_downcast.weight" in state_dict:
        model.speech_group_downcast.kernel.value = _load_tensor(state_dict, "speech_group_downcast.weight",
                                                                transpose=True)
        print("Loaded speech_group_downcast")

    if "speech_embeddings_to_local.weight" in state_dict and model.speech_embeddings_to_local:
        model.speech_embeddings_to_local.kernel.value = _load_tensor(state_dict, "speech_embeddings_to_local.weight",
                                                                     transpose=True)
        print("Loaded speech_embeddings_to_local")

    print("MiMo Audio weights loaded successfully!")
