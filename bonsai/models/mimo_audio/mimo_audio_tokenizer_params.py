"""Weight loading for MiMo Audio Tokenizer from SafeTensors"""

import jax.numpy as jnp
from safetensors import safe_open
from typing import Dict


def load_tokenizer_weights_from_safetensors(model, safetensors_path: str):
    state_dict = {}
    with safe_open(safetensors_path, framework="numpy") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    _load_encoder_weights(model.encoder, state_dict)

    _load_decoder_weights(model.decoder, state_dict)

    print("Tokenizer weights loaded successfully!")


def _assign_linear(linear_module, weight_key: str, bias_key: str, state_dict: Dict):
    if weight_key in state_dict:
        w = jnp.array(state_dict[weight_key].T, dtype=linear_module.kernel.value.dtype)
        linear_module.kernel.value = w

    if hasattr(linear_module, 'bias') and linear_module.bias is not None and bias_key in state_dict:
        b = jnp.array(state_dict[bias_key], dtype=linear_module.bias.value.dtype)
        linear_module.bias.value = b


def _assign_conv1d(conv_module, weight_key: str, bias_key: str, state_dict: Dict):
    if weight_key in state_dict:
        # PyTorch Conv1d weight: [out_channels, in_channels, kernel_size]
        # JAX expects: [kernel_size, in_channels, out_channels]
        w = jnp.array(state_dict[weight_key].transpose(2, 1, 0), dtype=conv_module.kernel.value.dtype)
        conv_module.kernel.value = w

    if hasattr(conv_module, 'bias') and conv_module.bias is not None and bias_key in state_dict:
        b = jnp.array(state_dict[bias_key], dtype=conv_module.bias.value.dtype)
        conv_module.bias.value = b


def _assign_norm(norm_module, weight_key: str, bias_key: str, state_dict: Dict):
    if weight_key in state_dict:
        if hasattr(norm_module, 'scale'):
            w = jnp.array(state_dict[weight_key], dtype=norm_module.scale.value.dtype)
            norm_module.scale.value = w
        elif hasattr(norm_module, 'weight'):
            w = jnp.array(state_dict[weight_key], dtype=norm_module.weight.value.dtype)
            norm_module.weight.value = w

    if hasattr(norm_module, 'bias') and norm_module.bias is not None and bias_key in state_dict:
        b = jnp.array(state_dict[bias_key], dtype=norm_module.bias.value.dtype)
        norm_module.bias.value = b



def _load_encoder_weights(encoder, state_dict: Dict):
    prefix = "encoder."

    # Conv layers
    _assign_conv1d(encoder.conv1, prefix + "conv1.weight", prefix + "conv1.bias", state_dict)
    _assign_conv1d(encoder.conv2, prefix + "conv2.weight", prefix + "conv2.bias", state_dict)

    # Transformer layers
    for idx, layer in enumerate(encoder.layers):
        layer_prefix = f"{prefix}layers.{idx}."

        # Attention
        attn_prefix = layer_prefix + "self_attn."
        _assign_linear(layer.self_attn.q_proj, attn_prefix + "q_proj.weight", attn_prefix + "q_proj.bias", state_dict)
        _assign_linear(layer.self_attn.k_proj, attn_prefix + "k_proj.weight", attn_prefix + "k_proj.bias", state_dict)
        _assign_linear(layer.self_attn.v_proj, attn_prefix + "v_proj.weight", attn_prefix + "v_proj.bias", state_dict)
        _assign_linear(layer.self_attn.out_proj, attn_prefix + "out_proj.weight", attn_prefix + "out_proj.bias", state_dict)

        # Norms
        _assign_norm(layer.self_attn_layer_norm, layer_prefix + "self_attn_layer_norm.weight",
                     layer_prefix + "self_attn_layer_norm.bias", state_dict)
        _assign_norm(layer.final_layer_norm, layer_prefix + "final_layer_norm.weight",
                     layer_prefix + "final_layer_norm.bias", state_dict)

        # FFN
        _assign_linear(layer.fc1, layer_prefix + "fc1.weight", layer_prefix + "fc1.bias", state_dict)
        _assign_linear(layer.fc2, layer_prefix + "fc2.weight", layer_prefix + "fc2.bias", state_dict)

    # Layer norm
    _assign_norm(encoder.layer_norm, prefix + "layer_norm.weight", prefix + "layer_norm.bias", state_dict)

    # Downsample layer
    if encoder.down_sample_layer is not None:
        _assign_conv1d(encoder.down_sample_layer, prefix + "down_sample_layer.0.weight",
                       prefix + "down_sample_layer.0.bias", state_dict)
        _assign_norm(encoder.down_norm, prefix + "down_sample_norm.weight",
                     prefix + "down_sample_norm.bias", state_dict)

    # Quantizer
    if encoder.quantizer is not None:
        for idx in range(encoder.quantizer.n_q):
            key = f"{prefix}quantizer.vq.layers.{idx}._codebook.embed"
            if key in state_dict:
                embed = jnp.array(state_dict[key], dtype=encoder.quantizer.codebooks[idx].value.dtype)
                encoder.quantizer.codebooks[idx].value = embed

    print(f"Loaded encoder weights")


def _load_decoder_weights(decoder, state_dict: Dict):
    prefix = "decoder."

    if decoder.dconv1 is not None:
        _load_causal_conv_transpose(decoder.dconv1, prefix + "dconv1.", state_dict)

    for idx, layer in enumerate(decoder.layers):
        layer_prefix = f"{prefix}layers.{idx}."

        # Attention
        attn_prefix = layer_prefix + "self_attn."
        _assign_linear(layer.self_attn.q_proj, attn_prefix + "q_proj.weight", attn_prefix + "q_proj.bias", state_dict)
        _assign_linear(layer.self_attn.k_proj, attn_prefix + "k_proj.weight", attn_prefix + "k_proj.bias", state_dict)
        _assign_linear(layer.self_attn.v_proj, attn_prefix + "v_proj.weight", attn_prefix + "v_proj.bias", state_dict)
        _assign_linear(layer.self_attn.out_proj, attn_prefix + "out_proj.weight", attn_prefix + "out_proj.bias", state_dict)

        # Norms
        _assign_norm(layer.self_attn_layer_norm, layer_prefix + "self_attn_layer_norm.weight",
                     layer_prefix + "self_attn_layer_norm.bias", state_dict)
        _assign_norm(layer.final_layer_norm, layer_prefix + "final_layer_norm.weight",
                     layer_prefix + "final_layer_norm.bias", state_dict)

        # FFN
        _assign_linear(layer.fc1, layer_prefix + "fc1.weight", layer_prefix + "fc1.bias", state_dict)
        _assign_linear(layer.fc2, layer_prefix + "fc2.weight", layer_prefix + "fc2.bias", state_dict)

    # Layer norm
    _assign_norm(decoder.layer_norm, prefix + "layer_norm.weight", prefix + "layer_norm.bias", state_dict)

    # Downsample conv
    _load_causal_conv_transpose(decoder.dconv2, prefix + "dconv2.", state_dict)

    # Vocoder
    _load_vocoder_weights(decoder.vocoder, prefix + "vocoder.", state_dict)

    print(f"Loaded decoder weights")


def _load_causal_conv_transpose(module, prefix: str, state_dict: Dict):
    if prefix + "conv.weight" in state_dict:
        w = jnp.array(state_dict[prefix + "conv.weight"], dtype=module.conv.kernel.value.dtype)
        module.conv.kernel.value = w

    if prefix + "conv.bias" in state_dict:
        b = jnp.array(state_dict[prefix + "conv.bias"], dtype=module.conv.bias.value.dtype)
        module.conv.bias.value = b

    # GroupNorm
    _assign_norm(module.norm, prefix + "norm.weight", prefix + "norm.bias", state_dict)


def _load_vocoder_weights(vocoder, prefix: str, state_dict: Dict):
    _assign_linear(vocoder.embeddings, prefix + "embeddings.weight", prefix + "embeddings.bias", state_dict)

    for idx, layer in enumerate(vocoder.layers):
        layer_prefix = f"{prefix}layers.{idx}."

        # Attention
        attn_prefix = layer_prefix + "self_attn."
        _assign_linear(layer.self_attn.q_proj, attn_prefix + "q_proj.weight", attn_prefix + "q_proj.bias", state_dict)
        _assign_linear(layer.self_attn.k_proj, attn_prefix + "k_proj.weight", attn_prefix + "k_proj.bias", state_dict)
        _assign_linear(layer.self_attn.v_proj, attn_prefix + "v_proj.weight", attn_prefix + "v_proj.bias", state_dict)
        _assign_linear(layer.self_attn.out_proj, attn_prefix + "out_proj.weight", attn_prefix + "out_proj.bias", state_dict)

        # Norms
        _assign_norm(layer.self_attn_layer_norm, layer_prefix + "self_attn_layer_norm.weight",
                     layer_prefix + "self_attn_layer_norm.bias", state_dict)
        _assign_norm(layer.final_layer_norm, layer_prefix + "final_layer_norm.weight",
                     layer_prefix + "final_layer_norm.bias", state_dict)

        # FFN
        _assign_linear(layer.fc1, layer_prefix + "fc1.weight", layer_prefix + "fc1.bias", state_dict)
        _assign_linear(layer.fc2, layer_prefix + "fc2.weight", layer_prefix + "fc2.bias", state_dict)

    # Layer norm
    _assign_norm(vocoder.layer_norm, prefix + "layer_norm.weight", prefix + "layer_norm.bias", state_dict)

    # ISTFT head
    _assign_linear(vocoder.head.linear, prefix + "head.out.weight", prefix + "head.out.bias", state_dict)

    window_key = prefix + "head.istft.window"
    if window_key in state_dict:
        vocoder.head.istft.window = jnp.array(state_dict[window_key], dtype=vocoder.head.istft.window.dtype)
