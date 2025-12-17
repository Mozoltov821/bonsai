# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import math

import jax
from flax import nnx
from jax import P
from jax import numpy as jnp
from jax.sharding import PartitionSpec, get_abstract_mesh
from jaxtyping import Array

from bonsai.models.qwen3.modeling import (
    Cache,
    LayerCache,
    MLP,
    RMSNorm,
    ShardingCfg,
    ShardingSpec,
    _generate_pos_embeddings,
    apply_rope,
    compute_positions_from_segment_ids,
    count_left_pads,
    count_right_pads,
    shard,
)

_K_MASK = jnp.finfo(jnp.bfloat16).min


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    vocab_size: int
    emb_dim: int
    mlp_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    rope_theta: int
    norm_eps: float
    tie_word_embeddings: bool
    shd_cfg: ShardingCfg = ShardingCfg.no_sharding()

    @classmethod
    def _from_param(cls, use_sharding: bool, **kwargs):
        if use_sharding:
            kwargs["shd_cfg"] = ShardingCfg.default()
        return cls(**kwargs)

    @classmethod
    def qwen2_0_5b(cls, use_sharding: bool = False):  # qwen2.5-0.5B
        return cls._from_param(
            use_sharding,
            num_layers=24,
            vocab_size=151936,
            emb_dim=896,
            mlp_dim=4864,
            num_heads=14,
            head_dim=64,
            num_kv_heads=2,
            norm_eps=1e-06,
            rope_theta=1000000,
            tie_word_embeddings=True,
        )

    @classmethod
    def qwen2_1_5b(cls, use_sharding: bool = False):  # qwen2-1.5B
        return cls._from_param(
            use_sharding,
            num_layers=28,
            vocab_size=151936,
            emb_dim=1536,
            mlp_dim=8960,
            num_heads=12,
            head_dim=128,
            num_kv_heads=2,
            norm_eps=1e-06,
            rope_theta=1000000,
            tie_word_embeddings=True,
        )


    @classmethod
    def qwen2_7b(cls, use_sharding: bool = False):  # qwen2-7B
        return cls._from_param(
            use_sharding,
            num_layers=28,
            vocab_size=151936,
            emb_dim=3584,
            mlp_dim=18944,
            num_heads=28,
            head_dim=128,
            num_kv_heads=4,
            norm_eps=1e-06,
            rope_theta=1000000,
            tie_word_embeddings=False,
        )

    @classmethod
    def qwen2_72b(cls, use_sharding: bool = False):  # qwen2-72B
        return cls._from_param(
            use_sharding,
            num_layers=80,
            vocab_size=151936,
            emb_dim=8192,
            mlp_dim=29568,
            num_heads=64,
            head_dim=128,
            num_kv_heads=8,
            norm_eps=1e-06,
            rope_theta=1000000,
            tie_word_embeddings=False,
        )


class Einsum(nnx.Module):
    def __init__(self, einsum_str: str, shape: tuple[int, ...], *, shd: ShardingSpec, rngs: nnx.Rngs, use_bias: bool = False):
        self.einsum_str = einsum_str
        self.shape = shape
        self.use_bias = use_bias
        self.w = shard(nnx.Param(nnx.initializers.normal()(rngs.params(), shape)), shd)
        if use_bias:
            # For linear layers, bias shape is the output dimension (last dim of weight)
            bias_shape = (shape[-1],)
            self.bias = shard(nnx.Param(nnx.initializers.zeros_init()(rngs.params(), bias_shape)), shd[-1:])

    @jax.named_scope("einsum")
    def __call__(self, x: Array) -> Array:
        result = jnp.einsum(self.einsum_str, x, self.w.value)
        if self.use_bias and hasattr(self.bias, 'value'):
            result = result + self.bias.value
        return result


class Attention(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.shd_cfg = cfg.shd_cfg

        # Use standard Linear layers (bias will be added manually)
        self.q_proj = nnx.Linear(cfg.emb_dim, cfg.emb_dim, use_bias=True, rngs=rngs)
        self.k_proj = nnx.Linear(cfg.emb_dim, cfg.num_kv_heads * cfg.head_dim, use_bias=True, rngs=rngs)
        self.v_proj = nnx.Linear(cfg.emb_dim, cfg.num_kv_heads * cfg.head_dim, use_bias=True, rngs=rngs)
        self.o_proj = nnx.Linear(cfg.num_heads * cfg.head_dim, cfg.emb_dim, use_bias=False, rngs=rngs)

        self.cfg = cfg
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.scale = cfg.head_dim**-0.5
        self.rope_theta = cfg.rope_theta

    @jax.named_scope("attention")
    def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array) -> Array:
        # Linear projections
        q_linear = self.q_proj(x)
        k_linear = self.k_proj(x)
        v_linear = self.v_proj(x)
        b, t, _ = q_linear.shape
        query_proj = q_linear.reshape(b, t, self.cfg.num_heads, self.cfg.head_dim)  # [B, T, N, H]
        key_proj = k_linear.reshape(b, t, self.cfg.num_kv_heads, self.cfg.head_dim)  # [B, T, K, H]
        value_proj = v_linear.reshape(b, t, self.cfg.num_kv_heads, self.cfg.head_dim)  # [B, T, K, H]

        # Apply sharding
        query_proj = shard(query_proj, self.shd_cfg.act_btnh)
        key_proj = shard(key_proj, self.shd_cfg.act_btnh)
        value_proj = shard(value_proj, self.shd_cfg.act_btnh)

        # RoPE and Cache Logic
        left_pads = count_left_pads(segment_ids)
        left_pads = shard(left_pads, P(self.shd_cfg.act_btnh[0]))
        cache.start_ind.value = jnp.where(cache.start_ind.value < 0, left_pads, cache.start_ind.value)
        position_ids = compute_positions_from_segment_ids(segment_ids) + cache.cur_ind.value
        sin, cos = _generate_pos_embeddings(position_ids, self.head_dim, self.rope_theta)
        query_proj = apply_rope(query_proj, sin, cos)
        key_proj = apply_rope(key_proj, sin, cos)

        # Ensure dtype matches cache (convert to cache dtype)
        cache_dtype = cache.k_cache.dtype
        value_proj = value_proj.astype(cache_dtype)
        key_proj = key_proj.astype(cache_dtype)

        # Update K/V cache [B, S, K, H]
        slice_indices = (0, cache.cur_ind.value, 0, 0)
        cache.v_cache.value = jax.lax.dynamic_update_slice(cache.v_cache.value, value_proj, slice_indices)
        cache.k_cache.value = jax.lax.dynamic_update_slice(cache.k_cache.value, key_proj, slice_indices)

        b, t, n, h = query_proj.shape

        # GQA reshape and attention logits
        query_proj_gqa = query_proj.reshape((b, t, self.num_kv_heads, self.n_rep, h))
        attn_logits = jnp.einsum("BTKGH,BSKH->BTSKG", query_proj_gqa, cache.k_cache.value) * self.scale

        # Masking and Softmax
        q_pos = cache.cur_ind.value + jnp.arange(t, dtype=jnp.int32)[None, :] - cache.start_ind.value[:, None]
        ts = jnp.arange(cache.size, dtype=jnp.int32)  # (cache.size,)
        kv_segment_ids = (ts[None, :] >= cache.start_ind.value[:, None]) & (ts[None, :] < cache.cur_ind.value + t)
        k_pos = ts[None, :] - cache.start_ind.value[:, None]  # (b, cache.size)
        causal_mask = k_pos[:, None, :] <= q_pos[:, :, None]
        segment_mask = kv_segment_ids[:, None, :] == segment_ids[:, :, None]
        final_mask = causal_mask & segment_mask  # (B, T, S)
        attn_mask = final_mask[:, :, :, None, None]
        attn_logits = jnp.where(attn_mask, attn_logits, _K_MASK)

        # Softmax
        attn_weights = jax.nn.softmax(attn_logits.astype(jnp.float32), axis=2).astype(attn_logits.dtype)
        qkv = jnp.einsum("BTSKG,BSKH->BTKGH", attn_weights, cache.v_cache.value)
        qkv = qkv.reshape((b, t, n, h))

        # Reshape for o_proj: (B, T, N, H) -> (B, T, N*H)
        qkv = qkv.reshape((b, t, n * h))

        output = self.o_proj(qkv)

        cache.cur_ind.value = cache.cur_ind.value + t
        return shard(output, self.shd_cfg.act_btd)

    @property
    def head_dim(self):
        return self.cfg.head_dim

    @property
    def num_heads(self):
        return self.cfg.num_heads

    @property
    def num_kv_heads(self):
        return self.cfg.num_kv_heads


class DecoderLayer(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.input_layernorm = RMSNorm(cfg.emb_dim, cfg, rngs=rngs)
        self.attn = Attention(cfg=cfg, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(cfg.emb_dim, cfg, rngs=rngs)
        self.mlp = MLP(cfg=cfg, rngs=rngs)

    def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array) -> Array:
        inputs_normalized = self.input_layernorm(x)
        attn_output = x + self.attn(inputs_normalized, cache, segment_ids)
        outputs = attn_output + self.mlp(self.post_attention_layernorm(attn_output))
        return outputs


class Qwen2(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.embedder = shard(
            nnx.Embed(num_embeddings=cfg.vocab_size, features=cfg.emb_dim, dtype=jnp.bfloat16, rngs=rngs),
            cfg.shd_cfg.emb_vd,
        )
        self.out_emb_shd = None if get_abstract_mesh().empty else cfg.shd_cfg.act_btd
        self.layers = nnx.List([DecoderLayer(cfg=cfg, rngs=rngs) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.emb_dim, cfg, rngs=rngs)
        self.lm_head = Einsum(
            einsum_str="BTD,DV->BTV", shape=(cfg.emb_dim, cfg.vocab_size), shd=cfg.shd_cfg.emb_dv, rngs=rngs
        )

    def init_cache(
        self, cfg: ModelConfig, batch_size: int, token_len: int, generate_steps: int, dtype: jnp.dtype = jnp.bfloat16
    ) -> Cache:
        cache_size = 2 ** math.ceil(math.log2(max(token_len + generate_steps, 1)))
        return [LayerCache(cfg, batch_size, cache_size, dtype) for _ in range(cfg.num_layers)]

    @nnx.jit
    def forward(self, cache: Cache, tokens: Array, pad_id: int, vocab_limit: int = 151643) -> tuple[Array, Cache]:
        segment_ids = 1 * (tokens != pad_id)
        num_right_pads = count_right_pads(tokens, pad_id)
        logits = self(tokens, segment_ids, cache, num_right_pads)
        target_ind = tokens.shape[-1] - num_right_pads - 1

        mask = jnp.arange(logits.shape[-1]) >= vocab_limit
        logits = jnp.where(mask[None, :], -jnp.inf, logits)

        return logits[:, target_ind], cache

    def __call__(self, tokens, segment_ids, cache, num_right_pads):
        x = self.embedder.embedding.value.at[(tokens,)].get(out_sharding=self.out_emb_shd)
        for i, layer in enumerate(self.layers):
            x = layer(x, cache[i], segment_ids)
        logits = self.lm_head(self.final_norm(x))
        return logits


def forward(model: Qwen2, cache: Cache, tokens: Array, pad_id: int, vocab_limit: int = 151643) -> tuple[Array, Cache]:
    """Backward compatibility wrapper. Use model.forward() instead."""
    return model.forward(cache, tokens, pad_id, vocab_limit)