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

"""FastAPI server for Qwen2 JAX model with OpenAI-compatible API."""

import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from transformers import AutoTokenizer

from bonsai.models.qwen2 import modeling, params
from bonsai.utils import Sampler


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0  # Use greedy decoding for deterministic evaluation
    top_p: float = 0.9
    top_k: int = 20
    max_tokens: int = 2048
    stream: bool = False
    use_chat_template: bool = False  # Default to False for evaluation tasks (preserves few-shot format)


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    temperature: float = 0.0  # Use greedy decoding for deterministic evaluation
    top_p: float = 0.9
    top_k: int = 20
    max_tokens: int = 2048


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "bonsai"


class Qwen2Server:
    def __init__(self, model_path: str, model_size: str = "0.5b"):
        self.model_path = model_path
        self.model_size = model_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model config
        config_map = {
            "0.5b": modeling.ModelConfig.qwen2_0_5b,
            "1.5b": modeling.ModelConfig.qwen2_1_5b,
            "7b": modeling.ModelConfig.qwen2_7b,
            "72b": modeling.ModelConfig.qwen2_72b,
        }
        if model_size not in config_map:
            raise ValueError(f"Unsupported model size: {model_size}")

        self.config = config_map[model_size](use_sharding=False)
        self.model = params.create_model_from_safe_tensors(model_path, self.config)
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id

        # Get im_end token ID for chat completion
        # Qwen2 uses <|im_end|> to mark the end of assistant responses
        im_end_token = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        self.im_end_id = im_end_token[0] if im_end_token else None
        print(f"EOS token ID: {self.eos_id}")
        print(f"IM_END token ID: {self.im_end_id}")

        # JIT compile forward for better performance
        print("Compiling forward function with JIT...")

        # @jax.jit
        def forward_jit(model, cache, tokens, pad_id):
            return modeling.forward(model, cache, tokens, pad_id)

        self.forward_jit = forward_jit

        # Warmup: compile with dummy input
        print("Warming up model...")
        dummy_tokens = jnp.array([[1, 2, 3, 4, 5]])
        dummy_cache = self.model.init_cache(self.config, 1, 5, 10)
        _ = self.forward_jit(self.model, dummy_cache, dummy_tokens, self.pad_id)
        print("Model ready!")

    def tokenize(self, texts: list[str]) -> jnp.ndarray:
        """Tokenize input texts with left padding."""
        lines = [self.tokenizer.encode(text) for text in texts]
        max_len = max(len(line) for line in lines)
        return jnp.array(
            [np.pad(l, (max_len - len(l), 0), constant_values=self.pad_id) for l in lines]
        )

    def tokenize_chat(self, messages: list[ChatMessage], use_chat_template: bool = True) -> jnp.ndarray:
        """Tokenize chat messages using chat template."""
        chat_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        if use_chat_template:
            text = self.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
            # Debug: print the formatted prompt
            print("\n" + "="*80)
            print("🔍 DEBUG: Formatted prompt sent to model (with chat template):")
            print("="*80)
            print(text)
            print("="*80 + "\n")
        else:
            # For evaluation: just concatenate messages without chat template
            # This preserves the exact few-shot format
            text = "\n\n".join([msg.content for msg in messages])
            print("\n" + "="*80)
            print("🔍 DEBUG: Raw prompt sent to model (no chat template):")
            print("="*80)
            print(text)
            print("="*80 + "\n")

        return self.tokenize([text])

    def generate(
        self,
        tokens: jnp.ndarray,
        max_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 10,
        check_im_end: bool = True,  # New parameter to control im_end checking
    ) -> tuple[jnp.ndarray, int]:
        """Generate tokens using the model."""
        batch_size, token_len = tokens.shape
        cache = self.model.init_cache(self.config, batch_size, token_len, max_tokens)

        # Handle temperature=0 for greedy decoding
        if temperature == 0.0 or temperature < 1e-6:
            # Greedy decoding - just take argmax
            def greedy_sample(logits, key=None):
                return jnp.argmax(logits, axis=-1, keepdims=True)
            sampler_fn = greedy_sample
        else:
            sampler = Sampler(temperature=temperature, top_p=top_p, top_k=top_k)
            sampler_fn = jax.jit(sampler)

        key = jax.random.key(int(time.time() * 1000) % 2**32)

        # Prefill
        # logits, cache = self.model.forward(cache, tokens, self.pad_id)
        logits, cache = modeling.forward(self.model, cache, tokens, self.pad_id)
        next_tokens = sampler_fn(logits, key=key)

        # Decode
        tokens_list = [next_tokens]
        finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        for _ in range(max_tokens):
            logits, cache = modeling.forward(self.model, cache, next_tokens, self.pad_id)
            next_tokens = sampler_fn(logits, key=key)

            # Check for both EOS and IM_END tokens
            is_eos = next_tokens.squeeze(-1) == self.eos_id
            is_im_end = (
                (next_tokens.squeeze(-1) == self.im_end_id)
                if (check_im_end and self.im_end_id is not None)
                else jnp.zeros_like(finished)
            )
            finished = finished | is_eos | is_im_end

            tokens_list.append(next_tokens)
            if finished.all():
                break

        all_output_tokens = jax.device_get(jnp.concatenate(tokens_list, axis=-1))
        num_generated = all_output_tokens.shape[1]

        return all_output_tokens, num_generated


app = FastAPI(title="Qwen2 JAX API", version="1.0.0")
server: Optional[Qwen2Server] = None


@app.on_event("startup")
async def startup_event():
    global server
    import argparse
    import sys
    from pathlib import Path

    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2-0.5B",
                        help="Model path (local directory or HuggingFace repo ID)")
    parser.add_argument("--model-size", type=str, default="0.5b")
    args, _ = parser.parse_known_args()

    print(f"Loading model from {args.model_path}...")

    # Check if it's a local path or HuggingFace repo
    if Path(args.model_path).exists():
        # Local path
        model_path = args.model_path
        print(f"Using local model at: {model_path}")
    else:
        # HuggingFace repo ID - download it
        print(f"Downloading from HuggingFace: {args.model_path}")
        model_path = snapshot_download(args.model_path)
        print(f"Downloaded to: {model_path}")

    server = Qwen2Server(model_path, args.model_size)
    print("Model loaded successfully!")


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            ModelInfo(
                id=f"qwen2-{server.model_size}",
                created=int(time.time()),
            )
        ],
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint."""
    if server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Print incoming request
    print("\n" + "="*80)
    print("📥 CHAT COMPLETION REQUEST")
    print("="*80)
    for msg in request.messages:
        print(f"[{msg.role}]: {msg.content}")
    print(f"Temperature: {request.temperature}, Top-P: {request.top_p}, Top-K: {request.top_k}")
    print(f"Max tokens: {request.max_tokens}")
    print(f"Use chat template: {request.use_chat_template}")
    print("-"*80)

    try:
        # Tokenize
        tokens = server.tokenize_chat(request.messages, use_chat_template=request.use_chat_template)
        prompt_tokens = tokens.shape[1]

        # Generate
        output_tokens, num_generated = server.generate(
            tokens,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            check_im_end=request.use_chat_template,  # Only check im_end when using chat template
        )

        # Decode
        seq_tokens = output_tokens[0]

        # Find first occurrence of EOS or IM_END
        eos_idx = np.where(seq_tokens == server.eos_id)[0]
        im_end_idx = np.where(seq_tokens == server.im_end_id)[0] if server.im_end_id is not None else np.array([])

        # Use the earliest stopping token
        stop_indices = []
        if eos_idx.size > 0:
            stop_indices.append(eos_idx[0])
        if im_end_idx.size > 0:
            stop_indices.append(im_end_idx[0])

        if stop_indices:
            seq_tokens = seq_tokens[: min(stop_indices)]

        response_text = server.tokenizer.decode(seq_tokens, skip_special_tokens=True)
        completion_tokens = len(seq_tokens)
        stopped = len(stop_indices) > 0

        # Print response
        print("📤 CHAT COMPLETION RESPONSE")
        print("-"*80)
        print(f"[assistant]: {response_text}")
        print("-"*80)
        print(f"✅ Tokens generated: {completion_tokens}")
        print(f"   Prompt tokens: {prompt_tokens}")
        print(f"   Total tokens: {prompt_tokens + completion_tokens}")
        print(f"   Finish reason: {'stop' if stopped else 'length'}")
        print("="*80 + "\n")

        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time() * 1000)}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop" if stopped else "length",
                }
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions", response_model=CompletionResponse)
async def completion(request: CompletionRequest):
    """OpenAI-compatible text completion endpoint."""
    if server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Print incoming request
    print("\n" + "="*80)
    print("📥 TEXT COMPLETION REQUEST")
    print("="*80)
    prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i+1}: {prompt}")
    print(f"Temperature: {request.temperature}, Top-P: {request.top_p}, Top-K: {request.top_k}")
    print(f"Max tokens: {request.max_tokens}")
    print("-"*80)

    try:
        # Handle both single and batch prompts
        tokens = server.tokenize(prompts)
        prompt_tokens = tokens.shape[1]

        # Generate
        output_tokens, num_generated = server.generate(
            tokens,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )

        # Decode all responses
        choices = []
        for i, seq_tokens in enumerate(output_tokens):
            # Find first occurrence of EOS or IM_END
            eos_idx = np.where(seq_tokens == server.eos_id)[0]
            im_end_idx = np.where(seq_tokens == server.im_end_id)[0] if server.im_end_id is not None else np.array([])

            # Use the earliest stopping token
            stop_indices = []
            if eos_idx.size > 0:
                stop_indices.append(eos_idx[0])
            if im_end_idx.size > 0:
                stop_indices.append(im_end_idx[0])

            stopped = False
            if stop_indices:
                seq_tokens = seq_tokens[: min(stop_indices)]
                stopped = True

            response_text = server.tokenizer.decode(seq_tokens, skip_special_tokens=True)

            choices.append(
                {
                    "index": i,
                    "text": response_text,
                    "finish_reason": "stop" if stopped else "length",
                }
            )

        completion_tokens = sum(len(c["text"].split()) for c in choices)

        # Print response
        print("📤 TEXT COMPLETION RESPONSE")
        print("-"*80)
        for i, choice in enumerate(choices):
            print(f"Response {i+1}: {choice['text']}")
            print(f"  Finish reason: {choice['finish_reason']}")
        print("-"*80)
        print(f"✅ Total tokens generated: {completion_tokens}")
        print(f"   Prompt tokens: {prompt_tokens}")
        print(f"   Total tokens: {prompt_tokens + completion_tokens}")
        print("="*80 + "\n")

        return CompletionResponse(
            id=f"cmpl-{int(time.time() * 1000)}",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": server is not None}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Qwen2 JAX API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2-0.5B", help="Model path or HF repo")
    parser.add_argument("--model-size", type=str, default="0.5b", choices=["0.5b", "1.5b", "7b", "72b"])

    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
