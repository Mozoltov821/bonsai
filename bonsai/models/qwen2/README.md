# Qwen2 in JAX

This directory contains a pure JAX implementation of the [Qwen2 language model](https://qwenlm.github.io/blog/qwen2/), using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API.

## Model Configuration Support Status

| Model Name | Config Support Status |
| :--- | :--- |
| **Dense Models** | |
| [Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B) | **✅ Supported** |
| [Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B) | **✅ Supported** |
| [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B) | **✅ Supported** |
| [Qwen2-72B](https://huggingface.co/Qwen/Qwen2-72B) | **✅ Supported** |

## Architecture Highlights

- **RoPE Theta**: Uses RoPE theta value of 1,000,000 (same as Qwen3)
- **Group Query Attention (GQA)**: Efficient attention mechanism with different head configurations
- **SwiGLU Activation**: Modern activation function for better performance
- **RMSNorm**: Root Mean Square Layer Normalization
- **Sharding Support**: Built-in support for distributed training and inference

### Running this model

Run Qwen2 in action, implemented in pure JAX.

```sh
python3 -m bonsai.models.qwen2.tests.run_model
```

## Installation

Install with Qwen2 dependencies:

```bash
pip install -e ".[qwen2]"
```

Or install all model dependencies:

```bash
pip install -e ".[models]"
```

## Usage Example

```python
import jax
import jax.numpy as jnp
from flax import nnx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from bonsai.models.qwen2 import modeling, params

# Load model and tokenizer
model_path = snapshot_download("Qwen/Qwen2-0.5B")
config = modeling.ModelConfig.qwen2_0_5b(use_sharding=False)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = params.create_model_from_safe_tensors(model_path, config)

# Tokenize input
text = ["Hello, how are you?"]
tokens = tokenizer(text, return_tensors="pt", padding=True)
tokens = jnp.array(tokens["input_ids"])

# Initialize cache and run inference
batch_size, seq_len = tokens.shape
cache = model.init_cache(config, batch_size, seq_len, generate_steps=10)
logits, _ = modeling.forward(model, cache, tokens, tokenizer.pad_token_id)
```

## Model Configurations

The implementation supports all Qwen2 model sizes:

- **0.5B**: 24 layers, 896 hidden size, 14 attention heads, 2 key-value heads
- **1.5B**: 28 layers, 1536 hidden size, 12 attention heads, 2 key-value heads
- **7B**: 28 layers, 3584 hidden size, 28 attention heads, 4 key-value heads
- **72B**: 80 layers, 8192 hidden size, 64 attention heads, 8 key-value heads

All configurations use:
- **RoPE theta**: 1,000,000
- **Vocab size**: 151,936
- **Norm epsilon**: 1e-6
- **SwiGLU activation** in MLP layers

## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Test existing configs on different hardware configurations and report compatibility
* Add optimizations for specific hardware (TPU, multi-GPU setups)
* Improve numerical stability or performance optimizations
* Add support for additional Qwen2 variants or fine-tuned models

To contribute:
1. Follow the [Bonsai contributing guidelines](../../../CONTRIBUTING.md)
2. Test your changes with `python -m bonsai.models.qwen2.tests.run_model`
3. Run numerical validation tests with `python -m pytest bonsai/models/qwen2/tests/`
4. Ensure pre-commit hooks pass: `pre-commit run --all-files`

## Architecture Differences from Qwen3

- **RoPE Configuration**: Same RoPE theta (1,000,000) but without rope_scaling_factor or local_rope_theta
- **No Query/Key Normalization**: Qwen2 uses standard attention without additional normalization layers
- **Model Variants**: Qwen2 has 0.5B, 1.5B, 7B, 72B variants vs. Qwen3's different size progression
- **Performance**: Qwen3 achieves better efficiency per parameter, while Qwen2 provides proven reliability