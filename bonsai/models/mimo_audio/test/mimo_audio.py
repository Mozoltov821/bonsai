import os
import re
import time
import random
import jax
import jax.numpy as jnp
from flax import nnx
from typing import Union, Optional, List
import soundfile as sf
from process_speechdata import InputSegment, StreamingInputSegment
from bonsai.models.mimo_audio.melSpectrogram import MelSpectrogram
from bonsai.models.mimo_audio.mimo_audio_tokenizer_params import load_tokenizer_weights_from_safetensors
from bonsai.models.mimo_audio.params import create_model_with_weights

from templates import asr_en_templates, asr_zh_templates, tts_en_templates, tts_zh_templates
from bonsai.models.mimo_audio.modeling import (
    MiMoAudioArguments,
    MiMoAudioConfig,
    MiMoSampler,
    MiMoSamplerConfig,
)
from transformers import PretrainedConfig


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



def detect_language(text):
    if re.search(r'[\u4e00-\u9fff]', text):
        return 'zh'
    else:
        return 'en'


def resample_jax(audio: jnp.ndarray, orig_sr: int, target_sr: int) -> jnp.ndarray:
    """Simple linear resampling for audio."""
    if orig_sr == target_sr:
        return audio

    duration = len(audio) / orig_sr
    num_samples = int(duration * target_sr)

    # Linear interpolation
    x_old = jnp.linspace(0, 1, len(audio))
    x_new = jnp.linspace(0, 1, num_samples)
    return jnp.interp(x_new, x_old, audio)


class MimoAudioJAX:
    """JAX implementation of MimoAudio interface"""

    def __init__(
            self,
            model_path: str,
            mimo_audio_tokenizer_path: str,
            tokenizer_path: Optional[str] = None,
    ) -> None:
        """
        Initialize MimoAudio with JAX backend.

        Args:
            model_path: Path to the main model
            mimo_audio_tokenizer_path: Path to the audio tokenizer model
            tokenizer_path: Optional path to text tokenizer (defaults to model_path)
        """
        self.path = model_path
        self.mimo_audio_tokenizer_path = mimo_audio_tokenizer_path

        # Import tokenizer (still using HuggingFace for text tokenization)
        from transformers import AutoTokenizer
        tokenizer_path = tokenizer_path or self.path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.padding_idx = int(self.tokenizer.pad_token_id)

        # Add special tokens
        special_tokens = [
            "<|sosp|>", "<|eosp|>", "<|empty|>",
            "<|Human|>", "<|SpeechLM|>",
            "<|sostm|>", "<|eostm|>", "<|eot|>",
        ]
        for token in special_tokens:
            if token not in self.tokenizer.get_vocab():
                print(f"Add special tokens {token} to tokenizer.vocab")
                self.tokenizer.add_tokens([token], special_tokens=True)

        # Store special token indices
        self.sosp_idx = self.tokenizer.convert_tokens_to_ids("<|sosp|>")
        self.eosp_idx = self.tokenizer.convert_tokens_to_ids("<|eosp|>")
        self.empty_token = self.tokenizer.convert_tokens_to_ids("<|empty|>")
        self.sostm_idx = self.tokenizer.convert_tokens_to_ids("<|sostm|>")
        self.eostm_idx = self.tokenizer.convert_tokens_to_ids("<|eostm|>")
        self.eot_idx = self.tokenizer.convert_tokens_to_ids("<|eot|>")
        self.im_start_idx = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_idx = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        # Load main model
        start_loading_time = time.monotonic()

        # Load config
        import json
        with open(os.path.join(self.path, "config.json")) as f:
            config_dict = json.load(f)

        config = MiMoAudioConfig(**{k: v for k, v in config_dict.items() if k in MiMoAudioConfig.__dataclass_fields__})

        args = MiMoAudioArguments(
            model_name_or_path=self.path,
            sosp_idx=self.sosp_idx,
            eosp_idx=self.eosp_idx,
            empty_idx=self.empty_token,
            sostm_idx=self.sostm_idx,
            eostm_idx=self.eostm_idx,
            eot_idx=self.eot_idx,
        )

        self.model = create_model_with_weights(
            model_path=self.path,
            config=config,
            args=args,
            rngs=nnx.Rngs(0)
        )

        self.group_size = config.group_size
        self.audio_channels = config.audio_channels
        self.delay_pattern = config.parsed_delay_pattern()
        self.vocab_size = config.vocab_size
        self.speech_zeroemb_idx = config.parsed_speech_empty_ids()

        print(f"Model loaded in {time.monotonic() - start_loading_time:.2f} seconds")

        # Load audio tokenizer
        start_loading_mimo_audio_tokenizer_time = time.monotonic()

        tokenizer_config_path = os.path.join(self.mimo_audio_tokenizer_path, "config.json")
        with open(tokenizer_config_path) as f:
            tokenizer_config_dict = json.load(f)
        tokenizer_config = MiMoAudioTokenizerConfig(**tokenizer_config_dict)

        tokenizer_weights_path = os.path.join(self.mimo_audio_tokenizer_path, "model.safetensors")
        self.mimo_audio_tokenizer = load_tokenizer_weights_from_safetensors(
            tokenizer_config,
            tokenizer_weights_path,
            dtype=jnp.bfloat16,
            rngs=nnx.Rngs(0)
        )

        self.tokenizer_config = tokenizer_config
        print(f"MiMo-Audio Tokenizer loaded in {time.monotonic() - start_loading_mimo_audio_tokenizer_time:.2f} seconds")

        # Initialize mel spectrogram transform
        self.mel_transform = MelSpectrogram(
            sample_rate=tokenizer_config.sampling_rate,
            n_fft=tokenizer_config.nfft,
            hop_length=tokenizer_config.hop_length,
            win_length=tokenizer_config.window_size,
            f_min=tokenizer_config.fmin,
            f_max=tokenizer_config.fmax if tokenizer_config.fmax is not None else tokenizer_config.sampling_rate / 2.0,
            n_mels=tokenizer_config.n_mels,
            power=1.0,
            center=True,
        )

        # Default samplers
        self.default_global_sampler = MiMoSampler(MiMoSamplerConfig(
            do_sample=True, temperature=0.6, top_k=50, top_p=0.95
        ))
        self.default_local_sampler = MiMoSampler(MiMoSamplerConfig(
            do_sample=True, temperature=0.9, top_k=50, top_p=0.95
        ))

        # Task-specific samplers
        self.task_sampler_configs = {
            "asr": {
                "global": MiMoSampler(MiMoSamplerConfig(do_sample=False, temperature=1.0, top_p=1.0)),
                "local": MiMoSampler(MiMoSamplerConfig(do_sample=True, temperature=0.9, top_p=0.95))
            },
            "tts": {
                "global": MiMoSampler(MiMoSamplerConfig(do_sample=True, temperature=0.6, top_p=1.0)),
                "local": MiMoSampler(MiMoSamplerConfig(do_sample=True, temperature=0.9, top_p=0.95))
            },
            "spoken_dialogue": {
                "global": MiMoSampler(MiMoSamplerConfig(do_sample=True, temperature=0.6, top_p=0.95)),
                "local": MiMoSampler(MiMoSamplerConfig(do_sample=True, temperature=0.9, top_p=0.95))
            },
            "audio_understanding": {
                "global": MiMoSampler(MiMoSamplerConfig(do_sample=True, temperature=0.3, top_p=0.95)),
                "local": MiMoSampler(MiMoSamplerConfig(do_sample=True, temperature=0.9, top_p=0.95))
            },
            "text_chat": {
                "global": MiMoSampler(MiMoSamplerConfig(do_sample=True, temperature=0.4, top_p=0.95)),
                "local": MiMoSampler(MiMoSamplerConfig(do_sample=True, temperature=0.9, top_p=0.95))
            },
            "in_context_learning_s2s": {
                "global": MiMoSampler(MiMoSamplerConfig(do_sample=False, temperature=1.0, top_p=1.0)),
                "local": MiMoSampler(MiMoSamplerConfig(do_sample=True, temperature=0.9, top_p=0.95))
            },
        }

        self.history = None

    def get_task_sampler(self, task_name):
        """Get sampler configuration for a specific task"""
        if task_name not in self.task_sampler_configs:
            return {
                "global": self.default_global_sampler,
                "local": self.default_local_sampler
            }
        return self.task_sampler_configs[task_name]

    def save_wav(self, path, wav):
        """Save waveform to file"""
        wav_np = jnp.asarray(wav).reshape(-1)
        sf.write(path, wav_np, 24000)

    def resample_audio_if_needed(self, wav: jnp.ndarray, original_sr: int) -> jnp.ndarray:
        """Resample audio to target sample rate"""
        target_sr = self.tokenizer_config.sampling_rate
        if original_sr != target_sr:
            return resample_jax(wav, original_sr, target_sr)
        return wav

    def encode_batch(self, input_features: jnp.ndarray, input_lens: jnp.ndarray):
        """Encode audio features to tokens"""
        # Simple batching - for large inputs, split into chunks
        max_length = 256000
        total_length = int(jnp.sum(input_lens))

        if total_length <= max_length:
            codes_output = self.mimo_audio_tokenizer.encode(
                input_features[jnp.newaxis, :, :],
                input_lens[jnp.newaxis],
                use_quantizer=True
            )
            return codes_output.codes
        else:
            # Split into chunks
            encoded_parts = []
            current_pos = 0
            for length in input_lens:
                chunk = input_features[current_pos:current_pos+length, :]
                codes_output = self.mimo_audio_tokenizer.encode(
                    chunk[jnp.newaxis, :, :],
                    jnp.array([length]),
                    use_quantizer=True
                )
                encoded_parts.append(codes_output.codes)
                current_pos += length

            return jnp.concatenate(encoded_parts, axis=-1)

    def preprocess_input(self, input: Union[None, str, jnp.ndarray] = None):
        """Preprocess input (audio file/tensor or text)"""
        if isinstance(input, jnp.ndarray) or (isinstance(input, str) and os.path.isfile(input)):
            # Audio input
            if isinstance(input, jnp.ndarray):
                wav = input
                sr = self.tokenizer_config.sampling_rate
            else:
                # Load audio file
                wav, sr = sf.read(input)
                wav = jnp.array(wav)
                if wav.ndim == 2:
                    wav = jnp.mean(wav, axis=1)

            wav = self.resample_audio_if_needed(wav, sr)

            # Compute mel spectrogram
            mel = self.mel_transform(wav)  # Returns (n_mels, time)
            mel = jnp.log(jnp.maximum(mel, 1e-7)).T  # (time, n_mels)

            # Split into segments for encoding
            input_len = mel.shape[0]
            segment_size = 6000
            input_len_seg = [segment_size] * (input_len // segment_size)
            if input_len % segment_size > 0:
                input_len_seg.append(input_len % segment_size)

            codes_packed = self.encode_batch(
                mel,
                jnp.array(input_len_seg)
            )

            codes = codes_packed.T  # Transpose to (timesteps, channels)
            audio_codes = codes[:, :self.audio_channels]

            # Pad to multiple of group_size
            num_timesteps = audio_codes.shape[0]
            if num_timesteps % self.group_size != 0:
                padding_needed = self.group_size - (num_timesteps % self.group_size)
                last_tokens = audio_codes[-1:, :]
                padding_tokens = jnp.repeat(last_tokens, padding_needed, axis=0)
                audio_codes = jnp.concatenate([audio_codes, padding_tokens], axis=0)

            audio_tokenized = audio_codes.reshape(-1)
            return audio_tokenized
        else:
            # Text input
            text = input
            if text.isupper() or text.islower():
                text = text.capitalize()
            return text

    def get_input_ids(self, prompt):
        """Convert prompt segments to input_ids"""
        input_ids = [
            seg.to_input_id(
                self.tokenizer,
                self.group_size,
                self.audio_channels,
            )
            for seg in prompt
        ]
        input_ids_concat = jnp.concatenate([jnp.array(ids) for ids in input_ids], axis=1)
        return input_ids_concat

    # Simplified interface - only implement key methods
    def forward(
            self,
            input_ids,
            return_audio=False,
            output_audio_path=None,
            stop_token_ids=None,
            min_new_tokens=0,
            max_new_tokens=8192,
            add_history=False,
            task_name=None,
    ):
        """Forward pass with generation and decoding"""
        task_sampler = self.get_task_sampler(task_name)

        # Flatten input_ids: [audio_channels+1, T] -> [1, flattened]
        input_ids = input_ids.T.reshape(1, -1)

        if add_history and self.history is not None:
            input_ids = jnp.concatenate([self.history, input_ids], axis=1)

        prompt_length = input_ids.shape[1] // (self.audio_channels + 1)
        max_length = prompt_length // self.group_size + max_new_tokens
        min_length = prompt_length // self.group_size + min_new_tokens

        # Generate
        key = jax.random.PRNGKey(int(time.time()))
        generated_ids = self.model.generate(
            input_ids,
            key,
            max_length=max_length,
            min_length=min_length,
            global_sampler=task_sampler["global"],
            local_sampler=task_sampler["local"],
            stop_token_ids=stop_token_ids or [self.tokenizer.eos_token_id],
            pad_id=self.tokenizer.pad_token_id,
        )

        self.history = generated_ids
        generated_ids = generated_ids.reshape(-1, self.audio_channels + 1).T[:, prompt_length:]

        # Extract text
        text = generated_ids[0, ::self.group_size]
        # Remove last token (usually EOS)
        if text.shape[0] > 0:
            text = text[:-1]

        detokenized_text = self.tokenizer.decode(
            [int(t) for t in text],
            skip_special_tokens=False
        ).strip().replace("<|empty|>", "").replace("<|eot|>", "").replace("<|eostm|>", "")

        print("Text channel:\t", detokenized_text)

        if output_audio_path:
            return_audio = True

        if not return_audio:
            return detokenized_text

        # Extract audio
        sosp_idx_locations = jnp.where(text == self.sostm_idx)[0]
        eosp_idx_locations = jnp.where(text == self.eostm_idx)[0]

        if len(sosp_idx_locations) == 0:
            start_location = 0
        else:
            start_location = int(sosp_idx_locations[0]) * self.group_size + self.group_size

        if len(eosp_idx_locations) == 0:
            end_location = text.shape[0] * self.group_size
        else:
            end_location = int(eosp_idx_locations[0]) * self.group_size

        audio_sequence = generated_ids[:, start_location:end_location]
        speech_sequence = audio_sequence[1:]  # Skip text channel

        # Filter out empty tokens
        speech_zeroemb = self.speech_zeroemb_idx[0] if isinstance(self.speech_zeroemb_idx, list) else self.speech_zeroemb_idx
        mask = speech_sequence[0] != speech_zeroemb
        speech_sequence = speech_sequence[:, mask]

        if speech_sequence.shape[1] == 0:
            # No audio generated
            wav = jnp.zeros(24000)
            if output_audio_path:
                self.save_wav(output_audio_path, wav)
            return detokenized_text if not return_audio else wav

        # Decode audio
        speech_sequence = speech_sequence.T.flatten()
        codes = speech_sequence.reshape(-1, self.audio_channels).T

        # Decode in segments
        segment_len = 1500
        wav_list = []
        for start in range(0, codes.shape[-1], segment_len):
            segment_codes = codes[:, start:start + segment_len]
            wav_segment = self.mimo_audio_tokenizer.decode(segment_codes[jnp.newaxis, :, :])
            wav_list.append(wav_segment)

        wav_concat = jnp.concatenate(wav_list, axis=-1)

        if output_audio_path:
            self.save_wav(output_audio_path, wav_concat)
            return detokenized_text
        else:
            return wav_concat

    def asr_sft(self, audio):
        """Speech recognition"""
        audio_tokenized = self.preprocess_input(audio)

        template = random.choice(asr_zh_templates + asr_en_templates)

        lm_prompt = [
            InputSegment(
                text=f"<|im_start|>user\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                audio=audio_tokenized,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=template + f"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
        ]

        input_ids = self.get_input_ids(lm_prompt)
        return self.forward(
            input_ids,
            stop_token_ids=[self.tokenizer.eos_token_id, self.im_end_idx],
            task_name="asr"
        )

    def tts_sft(self, text, output_path, instruct=None, read_text_only=True, prompt_speech=None):
        """Text-to-speech synthesis"""
        if prompt_speech is not None:
            assistant_prompt_audio_token = self.preprocess_input(prompt_speech)
        else:
            assistant_prompt_audio_token = None

        if not read_text_only:
            text_input = self.preprocess_input(text)
            if assistant_prompt_audio_token is None:
                lm_prompt = [
                    InputSegment(
                        text="<|im_start|>system\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="你需要根据指定的风格指令和文本内容来生成语音。",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_start|>user\n{text_input}<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="<|im_start|>assistant\n<think>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]
            else:
                lm_prompt = [
                    InputSegment(
                        text="<|im_start|>system\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="你需要根据指定的风格指令和文本内容来生成和语音prompt具有相同音色的语音。你的音色应该是：",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="",
                        audio=assistant_prompt_audio_token,
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_start|>user\n{text_input}<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="<|im_start|>assistant\n<think>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]
        else:
            language = detect_language(text)
            if language == "zh":
                template = random.choice(tts_zh_templates)
            else:
                template = random.choice(tts_en_templates)

            text_input = self.preprocess_input(text)
            if instruct is None:
                lm_prompt = [
                    InputSegment(
                        text=f"<|im_start|>user\n{template}: {text_input}<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="<|im_start|>assistant\n<|sostm|>",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]
            else:
                if assistant_prompt_audio_token is None:
                    lm_prompt = [
                        InputSegment(
                            text="<|im_start|>system\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text="你需要根据指定的风格指令和文本内容来生成语音。",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text="<|im_end|>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text=f"<|im_start|>user\n{template}: {text_input}({instruct})<|im_end|>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text="<|im_start|>assistant\n<think>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                    ]
                else:
                    lm_prompt = [
                        InputSegment(
                            text="<|im_start|>system\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text="你需要根据指定的风格指令和文本内容来生成和语音prompt具有相同音色的语音。你的音色应该是：",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text="",
                            audio=assistant_prompt_audio_token,
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text="<|im_end|>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text=f"<|im_start|>user\n{template}: {text_input}({instruct})<|im_end|>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text="<|im_start|>assistant\n<think>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                    ]

        input_ids = self.get_input_ids(lm_prompt)
        return self.forward(
            input_ids,
            output_audio_path=output_path,
            stop_token_ids=[self.tokenizer.eos_token_id, self.eostm_idx, self.im_end_idx],
            task_name="tts"
        )

    def spoken_dialogue_sft(self, input_speech, output_audio_path=None, system_prompt=None, prompt_speech=None, add_history=False):
        """Spoken dialogue"""
        audio_tokenized = self.preprocess_input(input_speech)

        lm_prompt = []

        if add_history and self.history is not None:
            lm_prompt += [
                InputSegment(
                    text="<|im_start|>user\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    audio=audio_tokenized,
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text="<|im_end|>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text="<|im_start|>assistant\n<|sostm|>",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
            ]
        else:
            if prompt_speech:
                lm_prompt += [
                    InputSegment(
                        text="<|im_start|>system\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="Your voice should be:",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        audio=self.preprocess_input(prompt_speech),
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]

            lm_prompt += [
                InputSegment(
                    text="<|im_start|>user\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            ]

            if system_prompt:
                lm_prompt += [
                    InputSegment(
                        text=system_prompt,
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="\n\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                ]

            lm_prompt += [
                InputSegment(
                    audio=audio_tokenized,
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text="<|im_end|>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text="<|im_start|>assistant\n<|sostm|>",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
            ]

        input_ids = self.get_input_ids(lm_prompt)
        return self.forward(
            input_ids,
            output_audio_path=output_audio_path,
            stop_token_ids=[self.tokenizer.eos_token_id, self.eostm_idx, self.im_end_idx],
            task_name="spoken_dialogue",
            add_history=add_history
        )

    def audio_understanding_sft(self, input_speech, input_text, thinking=False):
        """Audio understanding"""
        audio_tokenized = self.preprocess_input(input_speech)

        lm_prompt = [
            InputSegment(
                text="<|im_start|>user\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                audio=audio_tokenized,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=input_text,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text="<|im_end|>\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text="<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
        ]

        if not thinking:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n\n</think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        else:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )

        input_ids = self.get_input_ids(lm_prompt)
        return self.forward(
            input_ids,
            stop_token_ids=[self.tokenizer.eos_token_id, self.im_end_idx],
            task_name="audio_understanding"
        )

    def speech2text_dialogue_sft(self, input_speech, thinking=False, add_history=False):
        """Speech-to-text dialogue"""
        audio_tokenized = self.preprocess_input(input_speech)

        lm_prompt = [
            InputSegment(
                text="<|im_start|>user\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                audio=audio_tokenized,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text="<|im_end|>\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text="<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        ]

        if not thinking:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n\n</think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        else:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )

        input_ids = self.get_input_ids(lm_prompt)
        return self.forward(
            input_ids,
            stop_token_ids=[self.tokenizer.eos_token_id, self.im_end_idx],
            task_name="spoken_dialogue",
            add_history=add_history
        )

    def text_dialogue_sft(self, input_text, thinking=False, add_history=False):
        """Text dialogue"""
        lm_prompt = [
            InputSegment(
                text="<|im_start|>user\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=input_text,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text="<|im_end|>\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text="<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
        ]

        if not thinking:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n\n</think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        else:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )

        input_ids = self.get_input_ids(lm_prompt)
        return self.forward(
            input_ids,
            stop_token_ids=[self.tokenizer.eos_token_id, self.im_end_idx],
            task_name="text_chat",
            add_history=add_history
        )

    def spoken_dialogue_sft_multiturn(self, message_list, output_audio_path=None, system_prompt=None, prompt_speech=None):
        """Multi-turn spoken dialogue"""
        lm_prompt = []

        if prompt_speech:
            lm_prompt += [
                InputSegment(
                    text="<|im_start|>system\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text="Your voice should be:",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    audio=self.preprocess_input(prompt_speech),
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text="<|im_end|>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            ]

        for i in range(len(message_list)):
            if message_list[i]['role'] == 'user':
                lm_prompt += [
                    InputSegment(
                        text="<|im_start|>user\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                ]
                if system_prompt and i == 0:
                    lm_prompt += [
                        InputSegment(
                            text=system_prompt,
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text="\n\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        )
                    ]
                lm_prompt += [
                    InputSegment(
                        audio=self.preprocess_input(message_list[i]['content']),
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                ]
            elif message_list[i]['role'] == 'assistant':
                lm_prompt += [
                    InputSegment(
                        text="<|im_start|>assistant\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    StreamingInputSegment(
                        text=message_list[i]['content']["text"],
                        audio=self.preprocess_input(message_list[i]['content']["audio"]),
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                        tokenizer=self.tokenizer,
                        group_size=self.group_size,
                        audio_channels=self.audio_channels,
                    ),
                    InputSegment(
                        text="<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                ]
            else:
                raise ValueError(f"Invalid role: {message_list[i]['role']}")

        lm_prompt += [
            InputSegment(
                text="<|im_start|>assistant\n<|sostm|>",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
        ]

        input_ids = self.get_input_ids(lm_prompt)
        return self.forward(
            input_ids,
            output_audio_path=output_audio_path,
            stop_token_ids=[self.tokenizer.eos_token_id, self.eostm_idx, self.im_end_idx],
            task_name="spoken_dialogue",
            add_history=False
        )

    def speech2text_dialogue_sft_multiturn(self, message_list, thinking=False):
        """Multi-turn speech-to-text dialogue"""
        lm_prompt = []
        for i in range(len(message_list)):
            if message_list[i]['role'] == 'user':
                lm_prompt += [
                    InputSegment(
                        text="<|im_start|>user\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        audio=self.preprocess_input(message_list[i]['content']),
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                ]
            elif message_list[i]['role'] == 'assistant':
                lm_prompt += [
                    InputSegment(
                        text="<|im_start|>assistant\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=message_list[i]['content'],
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                ]
            else:
                raise ValueError(f"Invalid role: {message_list[i]['role']}")

        lm_prompt.append(
            InputSegment(
                text="<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        )

        if not thinking:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n\n</think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        else:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )

        input_ids = self.get_input_ids(lm_prompt)
        return self.forward(
            input_ids,
            stop_token_ids=[self.tokenizer.eos_token_id, self.im_end_idx],
            task_name="spoken_dialogue",
            add_history=False
        )

    def text_dialogue_sft_multiturn(self, message_list, thinking=False):
        """Multi-turn text dialogue"""
        lm_prompt = []
        for i in range(len(message_list)):
            if message_list[i]['role'] == 'user':
                lm_prompt += [
                    InputSegment(
                        text="<|im_start|>user\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=message_list[i]['content'],
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                ]
            elif message_list[i]['role'] == 'assistant':
                lm_prompt += [
                    InputSegment(
                        text="<|im_start|>assistant\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=message_list[i]['content'],
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                ]
            else:
                raise ValueError(f"Invalid role: {message_list[i]['role']}")

        lm_prompt.append(
            InputSegment(
                text="<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        )

        if not thinking:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n\n</think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        else:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )

        input_ids = self.get_input_ids(lm_prompt)
        return self.forward(
            input_ids,
            stop_token_ids=[self.tokenizer.eos_token_id, self.im_end_idx],
            task_name="text_chat",
            add_history=False
        )

    def in_context_learning_s2s(self, instruction, prompt_examples, audio, max_new_tokens=None, output_audio_path=None):
        """In-context learning speech-to-speech"""
        prompt = [
            InputSegment(
                text=f"[Int]:{instruction}\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        ]

        for i in range(len(prompt_examples)):
            prompt += [
                InputSegment(
                    audio=self.preprocess_input(prompt_examples[i]["input_audio"]),
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text="\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                StreamingInputSegment(
                    text=prompt_examples[i]["output_transcription"],
                    audio=self.preprocess_input(prompt_examples[i]["output_audio"]),
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                    tokenizer=self.tokenizer,
                    group_size=self.group_size,
                    audio_channels=self.audio_channels,
                ),
                InputSegment(
                    text=" \n\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
            ]

        prompt += [
            InputSegment(
                audio=self.preprocess_input(audio),
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text="\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text="<|sostm|>",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
        ]

        input_ids = self.get_input_ids(prompt)
        self.forward(
            input_ids,
            output_audio_path=output_audio_path,
            stop_token_ids=[self.tokenizer.eos_token_id, self.eostm_idx],
            max_new_tokens=max_new_tokens or 8192,
            task_name="in_context_learning_s2s"
        )

    def clear_history(self):
        """Clear conversation history"""
        self.history = None
        print("History cleared")
