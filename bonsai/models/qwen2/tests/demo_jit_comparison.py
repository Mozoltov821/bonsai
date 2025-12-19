"""Demonstration of nnx.jit vs jax.jit differences."""

from flax import nnx
import jax
import jax.numpy as jnp


class StatefulModel(nnx.Module):
    """Model with mutable state (like KV cache)."""

    def __init__(self):
        self.cache = nnx.Variable(jnp.zeros((4, 4)))
        self.call_count = nnx.Variable(jnp.array(0))

    def compute(self, x):
        """Method that modifies internal state."""
        # Update cache
        self.cache.value = self.cache.value + x
        # Increment counter
        self.call_count.value = self.call_count.value + 1
        return self.cache.value.sum()


def demo():
    print("=" * 60)
    print("Comparing nnx.jit vs jax.jit with Stateful NNX Modules")
    print("=" * 60)

    # Test 1: nnx.jit (CORRECT for NNX modules)
    print("\n✅ Test 1: Using nnx.jit (Recommended)")
    print("-" * 60)

    model1 = StatefulModel()

    @nnx.jit
    def forward_nnx(model, x):
        return model.compute(x)

    print(f"Initial cache sum: {model1.cache.value.sum()}")
    print(f"Initial call count: {model1.call_count.value}")

    result1 = forward_nnx(model1, jnp.ones((4, 4)))
    print(f"\nAfter 1st call:")
    print(f"  Result: {result1}")
    print(f"  Cache sum: {model1.cache.value.sum()}")
    print(f"  Call count: {model1.call_count.value}")

    result2 = forward_nnx(model1, jnp.ones((4, 4)))
    print(f"\nAfter 2nd call:")
    print(f"  Result: {result2}")
    print(f"  Cache sum: {model1.cache.value.sum()}")
    print(f"  Call count: {model1.call_count.value}")

    print("\n✅ nnx.jit correctly handles state mutations!")

    # Test 2: jax.jit (INCORRECT for NNX modules with state)
    print("\n\n❌ Test 2: Using jax.jit (Will Fail)")
    print("-" * 60)

    model2 = StatefulModel()

    @jax.jit
    def forward_jax(model, x):
        return model.compute(x)

    print(f"Initial cache sum: {model2.cache.value.sum()}")
    print(f"Initial call count: {model2.call_count.value}")

    try:
        result = forward_jax(model2, jnp.ones((4, 4)))
        print(f"Result: {result}")
    except Exception as e:
        print(f"\n💥 Error: {type(e).__name__}")
        print(f"   {str(e)[:100]}...")
        print("\n❌ jax.jit cannot handle NNX Variable mutations!")

    # Test 3: Pure functions with jax.jit (CORRECT)
    print("\n\n✅ Test 3: jax.jit with pure functions (Correct Usage)")
    print("-" * 60)

    @jax.jit
    def pure_function(x, y):
        """Pure function without state - perfect for jax.jit."""
        return x * 2 + y

    result = pure_function(jnp.array(3.0), jnp.array(5.0))
    print(f"Result: {result}")
    print("✅ jax.jit works perfectly for pure functions!")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
1. nnx.jit:
   - ✅ Use for NNX Modules with mutable state (Variables)
   - ✅ Automatically handles state extraction and updates
   - ✅ Perfect for models with KV cache, counters, etc.

2. jax.jit:
   - ❌ Cannot handle NNX Variable mutations
   - ✅ Use for pure functions (no side effects)
   - ✅ Standard JAX compilation for stateless operations

Rule of thumb:
- NNX Module methods → use @nnx.jit
- Pure functions → use @jax.jit
    """)


if __name__ == "__main__":
    demo()
