- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.scaled_matmul

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.scaled_matmul.rst "Download source file")
-  .pdf

# jax.nn.scaled_matmul

## Contents

- [`scaled_matmul()`](#jax.nn.scaled_matmul)

# jax.nn.scaled_matmul[\#](#jax-nn-scaled-matmul "Link to this heading")

jax.nn.scaled_matmul(*lhs*, *rhs*, *lhs_scales*, *rhs_scales*, *preferred_element_type=\<class 'numpy.float32'\>*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L1268-L1375)[\#](#jax.nn.scaled_matmul "Link to this definition")  
Scaled matrix multiplication function.

Performs block-scaled matmul of a and b using a_scales and b_scales. The last dim is the contracting dim, and block size is inferred.

Mathematically, this operation is equivalent to:

    a_block_size = a.shape[-1] // a_scales.shape[-1]
    b_block_size = b.shape[-1] // b_scales.shape[-1]
    a_scaled = a * jnp.repeat(a_scales, a_block_size, axis=-1)
    b_scaled = b * jnp.repeat(b_scales, b_block_size, axis=-1)
    jnp.einsum('BMK,BNK->BMN', a_scaled, b_scaled)

Parameters:  
- **lhs** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – Operand a, shape (B, M, K).

- **rhs** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – Operand b, shape (B, N, K).

- **lhs_scales** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – Shape (B, M, K_a), where K % K_a == 0.

- **rhs_scales** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – Shape (B, N, K_b), where K % K_b == 0.

- **preferred_element_type** (*DTypeLike,* *optional*) – Defaults to jnp.float32.

Returns:  
Array of shape (B, M, N).

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

- We currently do not support user-defined precision for customizing the compute data type. It is fixed to jnp.float32.

- Block size is inferred as K // K_a for a and K // K_b for b.

- To use cuDNN with Nvidia Blackwell GPUs, inputs must match:

      # mxfp8
      a, b: jnp.float8_e4m3fn | jnp.float8_e5m2
      a_scales, b_scales: jnp.float8_e8m0fnu
      block_size: 32
      # nvfp4
      a, b: jnp.float4_e2m1fn
      a_scales, b_scales: jnp.float8_e4m3fn
      block_size: 16

Examples

Basic case:

    >>> a = jnp.array([1, 2, 3]).reshape((1, 1, 3))
    >>> b = jnp.array([4, 5, 6]).reshape((1, 1, 3))
    >>> a_scales = jnp.array([0.5]).reshape((1, 1, 1))
    >>> b_scales = jnp.array([0.5]).reshape((1, 1, 1))
    >>> scaled_matmul(a, b, a_scales, b_scales)  
    Array([[[8.]]], dtype=float32)

Using fused cuDNN call on Blackwell GPUs:

    >>> dtype = jnp.float8_e4m3fn
    >>> a = jax.random.normal(jax.random.PRNGKey(1), (3, 128, 64), dtype=dtype)
    >>> b = jax.random.normal(jax.random.PRNGKey(2), (3, 128, 64), dtype=dtype)
    >>> a_scales = jnp.ones((3, 128, 4), dtype=jnp.float8_e8m0fnu)
    >>> b_scales = jnp.ones((3, 128, 4), dtype=jnp.float8_e8m0fnu)
    >>> scaled_matmul(a, b, a_scales, b_scales)  

[](jax.nn.dot_product_attention.html "previous page")

previous

jax.nn.dot_product_attention

[](jax.nn.get_scaled_dot_general_config.html "next page")

next

jax.nn.get_scaled_dot_general_config

Contents

- [`scaled_matmul()`](#jax.nn.scaled_matmul)

By The JAX authors

© Copyright 2024, The JAX Authors.\
