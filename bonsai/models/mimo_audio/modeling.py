from typing import Optional, Tuple, List
import jax
import jax.numpy as jnp
from flax import nnx
from bonsai.models.qwen2.modeling import Qwen2, ModelConfig as Qwen2Config, Cache
from bonsai.models.qwen3.modeling import shard
from bonsai.utils.samplers import Sampler, GreedySampler
from bonsai.models.mimo_audio.mimo_audio_configuration import (
    MiMoAudioConfig,
    MiMoAudioArguments,
    MiMoSamplerConfig
)


class MiMoSampler:
    """Sampling utilities for generation"""

    def __init__(self, config: MiMoSamplerConfig):
        self.config = config
        # 根据config创建对应的utils sampler
        if config.do_sample:
            self._sampler = Sampler(
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p
            )
        else:
            self._sampler = GreedySampler()

    def sample(
            self,
            logits: jnp.ndarray,
            key: jax.random.PRNGKey,
            removed_tokens: Optional[List[int]] = None
    ) -> jnp.ndarray:
        """Sample next token from logits

        Wrapper around utils.samplers with removed_tokens support.
        """
        # 1. Handle removed_tokens (MiMo特有功能)
        if removed_tokens:
            for t in removed_tokens:
                logits = logits.at[:, t].set(-jnp.inf)

        # 2. Call utils sampler
        result = self._sampler(logits, key=key)  # [B, 1]

        # 3. Reshape to match original API: [B, 1] → [B]
        return result[:, 0]


class FlaxMiMoAudioForCausalLM(nnx.Module):
    def __init__(
            self,
            config: MiMoAudioConfig,
            args: MiMoAudioArguments,
            rngs: Optional[nnx.Rngs] = None,
            dtype: jnp.dtype = jnp.bfloat16,  # ✅ 恢复：默认bfloat16节省内存
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.config = config
        self.args = args
        self.shd_cfg = config.shd_cfg  # Store sharding config

        # Fixed speech configurations (based on actual model)
        self.speech_vocab_sizes = [1025, 1025, 129, 129, 129, 129, 129, 129]
        self.speech_empty_ids = [1024, 1024, 128, 128, 128, 128, 128, 128]
        self.delay_pattern = [0, 1, 2, 3, 4, 5, 6, 7]

        self.group_size = config.group_size
        self.audio_channels = config.audio_channels

        # Create Qwen2 configurations
        self.qwen2_config = config.create_qwen2_config()
        self.local_qwen2_config = config.create_local_qwen2_config()
        self.input_local_qwen2_config = config.create_input_local_qwen2_config()

        # Initialize Qwen2 models
        self.model = Qwen2(self.qwen2_config, rngs=rngs)
        self.local_transformer = Qwen2(self.local_qwen2_config, rngs=rngs)
        self.input_local_transformer = Qwen2(self.input_local_qwen2_config, rngs=rngs)

        # Keep the main model's embedder for text token embedding
        self.local_transformer.embedder = None
        self.input_local_transformer.embedder = None

        # Text LM head (note: not used, model.lm_head is used instead)
        self.lm_head = None

        # Local transformer LM heads for each audio channel
        self.local_transformer_lm_heads = nnx.List([
            shard(
                nnx.Linear(
                    config.local_dim,
                    self.speech_vocab_sizes[i],
                    use_bias=False,
                    dtype=dtype,  # ✅ 使用传入的dtype而不是硬编码bfloat16
                    rngs=rngs
                ),
                self.shd_cfg.emb_dv
            )
            for i in range(self.audio_channels)
        ])

        # Speech embeddings for each audio channel
        # ⚠️  关键：使用float32保持累加精度，避免tokens重复
        # PyTorch实现中embeddings累加也保持较高精度
        self.speech_embeddings = nnx.List([
            shard(
                nnx.Embed(
                    self.speech_vocab_sizes[i],
                    config.input_local_dim,
                    dtype=jnp.float32,  # 使用float32保持累加精度
                    rngs=rngs
                ),
                self.shd_cfg.emb_vd
            )
            for i in range(self.audio_channels)
        ])

        # Group downcast for combining speech groups
        self.speech_group_downcast = shard(
            nnx.Linear(
                config.input_local_dim * config.group_size,
                config.hidden_size,
                use_bias=False,
                dtype=jnp.bfloat16,
                rngs=rngs
            ),
            self.shd_cfg.ffw_weight_df
        )

        # Hidden states downcast for local transformer
        self.hidden_states_downcast = shard(
            nnx.Linear(
                config.hidden_size,
                config.local_dim,
                use_bias=False,
                dtype=jnp.bfloat16,
                rngs=rngs
            ),
            self.shd_cfg.ffw_weight_df
        )

    def apply_input_local_transformer(
            self,
            speech_embeddings: jnp.ndarray,  # [B, T_groups, group_size, hidden_size]
            cache: Optional[Cache] = None
    ) -> jnp.ndarray:
        """
        Apply input local transformer to speech embeddings.

        Args:
            speech_embeddings: [B, T_groups, group_size, hidden_size]
            cache: Optional KV cache (not used during prefill)

        Returns:
            Encoded embeddings [B, T_groups, group_size, hidden_size]
        """
        B, T_groups, group_size, hidden_size = speech_embeddings.shape

        # Ensure correct dtype (bfloat16 for efficiency)
        speech_embeddings = speech_embeddings.astype(jnp.bfloat16)

        # Process each group independently
        input_embeddings = speech_embeddings.reshape(B * T_groups, group_size, hidden_size)

        # Create segment IDs (all 1s for valid tokens)
        segment_ids = jnp.ones((B * T_groups, group_size), dtype=jnp.int32)

        # Create cache if not provided (for standalone processing)
        if cache is None:
            cache = self.input_local_transformer.init_cache(
                self.input_local_qwen2_config,
                B * T_groups,
                group_size,
                generate_steps=0,
                dtype=jnp.bfloat16
            )

        # Run through input local transformer layers
        x = input_embeddings
        for i, layer in enumerate(self.input_local_transformer.layers):
            x = layer(x, cache[i], segment_ids)
        x = self.input_local_transformer.final_norm(x)

        # Reshape back to original format
        encoded_embeddings = x.reshape(B, T_groups, group_size, hidden_size)

        return encoded_embeddings

    def _prepare_input_embeds(
            self,
            input_ids: jnp.ndarray,  # [B, audio_channels + 1, new_T]
            text_embed_fn
    ) -> jnp.ndarray:
        """
        Prepare input embeddings from interleaved text and speech tokens.

        Args:
            input_ids: [B, audio_channels + 1, new_T]
            text_embed_fn: Function to embed text tokens

        Returns:
            Combined embeddings [B, T_groups, hidden_size]
        """
        B = input_ids.shape[0]

        # Extract text and speech tokens
        text_input_ids = input_ids[:, 0, ::self.group_size]  # [B, T_groups]
        speech_input_ids = input_ids[:, 1:, :].reshape(
            B, self.audio_channels, -1, self.group_size
        ).transpose(0, 2, 1, 3)  # [B, T_groups, audio_channels, group_size]

        is_speech = text_input_ids == self.args.empty_idx  # [B, T_groups]

        # Initialize speech embeddings
        speech_embeds = jnp.zeros(
            (B, is_speech.shape[1], self.group_size, self.config.input_local_dim),
            dtype=jnp.bfloat16
        )

        # Sum embeddings from all audio channels
        for idx in range(self.audio_channels):
            cur_empty = self.speech_empty_ids[idx]
            cur_embed = self.speech_embeddings[idx]
            cur_speech_ids = speech_input_ids[:, :, idx, :]  # [B, T_groups, group_size]
            cur_speech_embeds = cur_embed(cur_speech_ids)  # [B, T_groups, group_size, hidden_size]

            # Mask out empty tokens
            cur_mask = cur_speech_ids == cur_empty
            cur_speech_embeds = cur_speech_embeds * ~cur_mask[..., None]

            speech_embeds = speech_embeds + cur_speech_embeds

        # Mask non-speech positions
        speech_embeds = speech_embeds * is_speech[:, :, None, None]

        # Apply input_local_transformer (pass None for cache during prefill)
        speech_embeds = self.apply_input_local_transformer(speech_embeds, cache=None)

        # ✅ 关键修复：apply_input_local_transformer 后再次 mask（与官方实现一致）
        speech_embeds = speech_embeds * is_speech[:, :, None, None]

        T_groups = speech_embeds.shape[1]
        # Flatten group dimension and project
        speech_grouped_embeds = self.speech_group_downcast(
            speech_embeds.reshape(B, T_groups, -1)
        )  # [B, T_groups, hidden_size]

        # Get text embeddings
        # ✅ 关键修复：处理 -100 padding tokens
        # 将 -100 替换为 0（或任何有效索引），因为我们会mask掉这些位置
        text_input_ids_safe = jnp.where(text_input_ids == -100, 0, text_input_ids)
        text_embeds = text_embed_fn(text_input_ids_safe)  # [B, T_groups, hidden_size]

        # Mask掉 empty_idx 和 -100 的位置
        text_zero_mask = (text_input_ids == self.args.empty_idx) | (text_input_ids == -100)
        text_embeds = text_embeds * ~text_zero_mask[..., None]

        output = text_embeds + speech_grouped_embeds
        return shard(output, self.shd_cfg.act_btd)

    def forward(
            self,
            input_ids: jnp.ndarray,  # [B, audio_channels + 1, new_T]
            cache: Cache,
            pad_id: int = 0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through the model.

        Args:
            input_ids: [B, audio_channels + 1, new_T]
            cache: KV cache for main transformer
            pad_id: Padding token ID

        Returns:
            text_logits: [B, 1, vocab_size]
            local_hidden_states: [B, 1, local_dim]
        """
        # Get text token IDs for embedding
        text_input_ids = input_ids[:, 0, ::self.group_size]  # [B, T_groups]

        # Get text embeddings using the main model's embedder
        def text_embed_fn(x):
            return self.model.embedder.embedding.value[x]

        # Prepare combined text+speech embeddings
        inputs_embeds = self._prepare_input_embeds(input_ids, text_embed_fn)

        # Create segment IDs
        # ✅ 关键修复：只mask -100 padding，empty_idx是有意义的token不应该被mask
        # 官方使用全1的attention_mask，不mask empty_idx位置
        # -100在::group_size采样后不应出现在text_input_ids中，但为了安全仍然检查
        B, T_groups, _ = inputs_embeds.shape
        segment_ids = 1 * (text_input_ids != -100)  # [B, T_groups] - 只mask -100，不mask empty_idx

        # Run through main transformer (ensure bfloat16)
        x = inputs_embeds.astype(jnp.bfloat16)
        for i, layer in enumerate(self.model.layers):
            x = layer(x, cache[i], segment_ids)
        hidden_states = self.model.final_norm(x)  # [B, T_groups, hidden_size]

        # Compute text logits
        text_logits = self.model.lm_head(hidden_states[:, -1:, :])  # [B, 1, vocab_size]

        # Downcast hidden states for local transformer
        local_hidden_states = self.hidden_states_downcast(
            hidden_states[:, -1:, :]
        )  # [B, 1, local_dim]

        return text_logits, local_hidden_states

    def local_forward(
            self,
            local_embeds: jnp.ndarray,  # [B, 1, local_dim]
            key: jax.random.PRNGKey,
            local_sampler: Optional[MiMoSampler] = None,
    ) -> jnp.ndarray:
        """
        Generate audio tokens for one group using local transformer.

        Args:
            local_embeds: [B, 1, local_dim]
            key: Random key for sampling
            local_sampler: Sampler configuration

        Returns:
            local_tokens: [B, group_size, audio_channels]
        """
        B = local_embeds.shape[0]
        delay_iters = self.group_size + max(self.delay_pattern)

        local_tokens = jnp.zeros(
            (B, self.group_size, self.audio_channels),
            dtype=jnp.int32
        )

        if local_sampler is None:
            local_sampler = MiMoSampler(MiMoSamplerConfig())

        # ✅ 关键修复：每次调用都创建新的 cache（与官方实现一致）
        # 官方实现：past_key_values = DynamicCache() 每次都是新的
        # token_len=1: 每次迭代只输入1个token
        # generate_steps: 总共需要delay_iters个位置，已有1个，还需delay_iters-1个
        cache = self.local_transformer.init_cache(
            self.local_qwen2_config,
            B,
            token_len=1,  # 每次只输入1个token
            generate_steps=delay_iters - 1,  # 还需要生成delay_iters-1个token的空间
            dtype=jnp.bfloat16,
        )

        # Create segment IDs
        segment_ids = jnp.ones((B, 1), dtype=jnp.int32)

        for t in range(delay_iters):
            # ✅ 使用 JIT 编译的 transformer forward（加速核心计算）
            # 必须返回 cache 以确保状态正确更新
            hidden_state, cache = _local_transformer_step_jit(
                self.local_transformer, local_embeds, cache, segment_ids
            )  # [B, 1, local_dim]

            # Reset embeddings for next iteration
            next_local_embeds = jnp.zeros_like(local_embeds)

            # Generate token for each channel based on delay pattern
            for idx in range(self.audio_channels):
                cur_start = self.delay_pattern[idx]
                cur_end = cur_start + self.group_size
                cur_empty = self.speech_empty_ids[idx]

                if cur_start <= t < cur_end:
                    # Compute logits for this channel
                    cur_lm_head = self.local_transformer_lm_heads[idx]
                    cur_logits = cur_lm_head(hidden_state[:, -1, :])  # [B, vocab_size]

                    # Sample token
                    key, subkey = jax.random.split(key)
                    cur_token = local_sampler.sample(
                        cur_logits,
                        subkey,
                        removed_tokens=[cur_empty]
                    )

                    # Store token
                    local_tokens = local_tokens.at[:, t - cur_start, idx].set(cur_token)

                    # Get embedding for next iteration
                    # Add sequence dimension: [B] -> [B, 1] before embedding
                    cur_input_embed = self.speech_embeddings[idx](cur_token[:, None])  # [B, 1, embed_dim]

                    next_local_embeds = next_local_embeds + cur_input_embed

            local_embeds = next_local_embeds

        return local_tokens

    def generate(
            self,
            input_ids: jnp.ndarray,
            key: jax.random.PRNGKey,
            max_length: int = 100,
            global_sampler: Optional[MiMoSampler] = None,
            local_sampler: Optional[MiMoSampler] = None,
            stop_token_ids: Optional[List[int]] = None,
            min_length: int = 0,
            pad_id: int = 0,
    ) -> jnp.ndarray:
        """
        Generate audio tokens autoregressively.

        Args:
            input_ids: Initial input tokens [B, (audio_channels + 1) * group_size * T]
            key: Random key
            max_length: Maximum number of groups to generate
            global_sampler: Sampler for text tokens
            local_sampler: Sampler for audio tokens
            stop_token_ids: Token IDs that trigger stopping
            min_length: Minimum length before allowing stop
            pad_id: Padding token ID for cache initialization

        Returns:
            Generated tokens [B, (audio_channels + 1) * group_size * T']
        """
        if global_sampler is None:
            global_sampler = MiMoSampler(MiMoSamplerConfig())
        if local_sampler is None:
            local_sampler = MiMoSampler(MiMoSamplerConfig())
        if stop_token_ids is None:
            stop_token_ids = []

        B = input_ids.shape[0]
        cur_len = input_ids.shape[1] // (self.group_size * (self.audio_channels + 1))

        # Initialize KV cache for main transformer
        # ✅ 关键修复：不再为 local_transformer 预先创建 cache
        # local_forward 会在内部创建自己的 cache（每次都是新的）
        token_len = cur_len
        generate_steps = max_length - cur_len

        main_cache = self.model.init_cache(
            self.qwen2_config, B, token_len, generate_steps, dtype=jnp.bfloat16
        )

        while cur_len < max_length:
            # Prepare model inputs
            # [B, (audio_channels + 1) * group_size * T] -> [B, audio_channels + 1, T * group_size]
            model_input_ids = input_ids.reshape(
                B, -1, (self.audio_channels + 1) * self.group_size
            ).transpose(0, 2, 1).reshape(B, self.audio_channels + 1, -1)

            # Forward pass with cache
            text_logits, local_hidden_states = self.forward(
                model_input_ids,
                main_cache,
                pad_id=pad_id
            )

            # Sample next text token
            key, subkey = jax.random.split(key)
            removed_tokens = list(stop_token_ids) if cur_len < min_length else None
            next_text_token = global_sampler.sample(
                text_logits[:, -1, :],
                subkey,
                removed_tokens=removed_tokens
            )

            # Check if should generate speech or use empty tokens
            if next_text_token[0] != self.args.empty_idx:
                # Use empty embeddings for speech - shape: [B, group_size, audio_channels]
                zero_embed_tensor = jnp.array(self.speech_empty_ids, dtype=jnp.int32)
                # Repeat empty IDs for each position in the group
                next_speech_tokens = jnp.tile(
                    zero_embed_tensor[None, None, :],  # [1, 1, audio_channels]
                    (B, self.group_size, 1)  # Repeat to [B, group_size, audio_channels]
                )
            else:
                # Generate speech tokens with local transformer
                key, subkey = jax.random.split(key)
                next_speech_tokens = self.local_forward(
                    local_hidden_states,
                    subkey,
                    local_sampler
                )

            # Combine text and speech tokens
            next_text_tokens = jnp.broadcast_to(
                next_text_token[:, None, None],
                (B, self.group_size, 1)
            )  # [B, group_size, 1]

            next_tokens = jnp.concatenate(
                [next_text_tokens, next_speech_tokens], axis=-1
            ).reshape(B, -1)  # [B, group_size * (audio_channels + 1)]

            # Append to input_ids
            input_ids = jnp.concatenate([input_ids, next_tokens], axis=-1)

            # Check stopping criteria
            if stop_token_ids and cur_len >= min_length:
                step = (self.audio_channels + 1) * self.group_size
                if input_ids.shape[1] >= step:
                    last_token = input_ids[0, -step]
                    if int(last_token) in stop_token_ids:
                        break

            cur_len += 1

        return input_ids


# ============================================================================
# JIT-compiled functions for fast inference
# ============================================================================

@jax.jit
def _local_transformer_step_jit(
    local_transformer: nnx.Module,
    local_embeds: jnp.ndarray,
    cache: Cache,
    segment_ids: jnp.ndarray,
) -> Tuple[jnp.ndarray, Cache]:
    """
    JIT-compiled single step of local transformer forward pass.

    This is a helper function to accelerate the inner loop of local_forward.
    Being a module-level function (not instance method) allows JAX to properly
    JIT compile it.

    Args:
        local_transformer: The local transformer module
        local_embeds: [B, 1, local_dim]
        cache: Cache for local transformer
        segment_ids: [B, 1]

    Returns:
        hidden_state: [B, 1, local_dim]
        cache: Updated cache (IMPORTANT for correct behavior)
    """
    x = local_embeds.astype(jnp.bfloat16)
    for i, layer in enumerate(local_transformer.layers):
        x = layer(x, cache[i], segment_ids)
    hidden_state = local_transformer.final_norm(x)
    return hidden_state, cache


@jax.jit
def forward_jit(
    model: FlaxMiMoAudioForCausalLM,
    input_ids: jnp.ndarray,
    cache: Cache,
    pad_id: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, Cache]:
    """
    JIT-compiled forward pass for fast inference.

    Similar to qwen2's forward function, this returns the cache to enable
    proper JAX tracing of stateful computations.

    Args:
        model: FlaxMiMoAudioForCausalLM instance
        input_ids: [B, audio_channels + 1, T * group_size]
        cache: Cache for KV storage
        pad_id: Padding token ID

    Returns:
        text_logits: [B, 1, vocab_size]
        local_hidden_states: [B, 1, local_dim]
        cache: Updated cache (for JAX tracing)
    """
    text_logits, local_hidden_states = model.forward(input_ids, cache, pad_id)
    return text_logits, local_hidden_states, cache


@jax.jit
def local_forward_jit(
    model: FlaxMiMoAudioForCausalLM,
    local_embeds: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """
    JIT-compiled local forward pass for audio generation.

    NOTE: This version uses greedy sampling (no sampler parameter for JIT simplicity).
    For temperature-based sampling, use model.local_forward() directly.

    Args:
        model: FlaxMiMoAudioForCausalLM instance
        local_embeds: [B, 1, local_dim]
        key: Random key (used if needed in future, currently greedy)

    Returns:
        audio_tokens: [B, group_size, audio_channels]
    """
    # Use greedy sampling for JIT-compiled version
    return model.local_forward(local_embeds, key, local_sampler=None)


# Example usage:
if __name__ == "__main__":
    # Create configuration
    config = MiMoAudioConfig()
    args = MiMoAudioArguments(
        model_name_or_path="mimo-audio",
        sosp_idx=151646,
        eosp_idx=151647,
        sostm_idx=151648,
        eostm_idx=151649,
        eot_idx=151643,
        empty_idx=151645,
    )

    # Create model
    model = FlaxMiMoAudioForCausalLM(config,args)

    print("Model created successfully!")
    print(f"Audio channels: {model.audio_channels}")
    print(f"Group size: {model.group_size}")
    print(f"Speech vocab sizes: {model.speech_vocab_sizes}")
