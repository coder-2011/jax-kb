- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.scaled_dot_general

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.scaled_dot_general.rst "Download source file")
-  .pdf

# jax.nn.scaled_dot_general

## Contents

- [`scaled_dot_general()`](#jax.nn.scaled_dot_general)

# jax.nn.scaled_dot_general[\#](#jax-nn-scaled-dot-general "Link to this heading")

jax.nn.scaled_dot_general(*lhs*, *rhs*, *dimension_numbers*, *preferred_element_type=\<class 'numpy.float32'\>*, *configs=None*, *implementation=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L1408-L1491)[\#](#jax.nn.scaled_dot_general "Link to this definition")  
Scaled dot general operation.

Performs a generalized dot product with block-scaled quantization on the lhs and rhs inputs. This operation extends lax.dot_general to support user-defined scaling configurations.

Essentially, the operation follows:

    a, a_scales = quantize(lhs, configs[0])
    b, b_scales = quantize(rhs, configs[1])
    c = jax.nn.scaled_matmul(a, b, a_scales, b_scales)

Parameters:  
- **lhs** (*ArrayLike*) – Input array.

- **rhs** (*ArrayLike*) – Input array.

- **dimension_numbers** (*DotDimensionNumbers*) – A tuple of two tuples specifying the contraction and batch dimensions: ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims)).

- **preferred_element_type** (*DTypeLike,* *optional*) – Output data type of the dot product. Defaults to jnp.float32. Other valid types include jnp.bfloat16 and jnp.float16.

- **configs** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") *of* *BlockScaleConfig,* *optional*) – Scaling configurations for lhs, rhs, and gradients. Users can obtain valid configurations via jax.nn.get_scaled_dot_general_config. Currently, nvfp4 and mxfp8 are supported. If None, falls back to lax.dot_general.

- **implementation** ([*Literal*](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")*\['cudnn'\]* *\|* *None*) – str (Deprecated) Backend selector, now ignored. The system chooses the backend automatically. Scheduled for removal in future releases.

Returns:  
The resulting tensor, with batch dimensions first, followed by non-contracting/non-batch dimensions of lhs, and then those of rhs.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.nn.scaled_matmul()`](jax.nn.scaled_matmul.html#jax.nn.scaled_matmul "jax.nn.scaled_matmul"): Scaled matmul function.

- [`jax.lax.dot_general()`](jax.lax.dot_general.html#jax.lax.dot_general "jax.lax.dot_general"): General dot product operator.

Notes

- Unlike nn.scaled_matmul, which assumes quantized low-precision inputs with explicit scaling factors, this operator takes high-precision inputs, applies quantization internally, and handles the backward pass.

Examples

Creating config for mxfp8:

    >>> configs = [jax.nn.get_scaled_dot_general_config('mxfp8')] * 3

Creating config for nvfp4:

    >>> global_scale = jnp.array([0.5], jnp.float32)
    >>> configs = [jax.nn.get_scaled_dot_general_config('nvfp4', global_scale)] * 3

Using scaled_dot_general with the configs:

    >>> import functools
    >>> scaled_dot_general_fn = functools.partial(jax.nn.scaled_dot_general, configs=configs)
    >>> lhs = jax.random.normal(jax.random.PRNGKey(1), (3, 128, 64))
    >>> rhs = jax.random.normal(jax.random.PRNGKey(2), (3, 128, 64))
    >>> out = scaled_dot_general_fn(lhs, rhs, (((2,), (2,)), ((0,), (0,))))  

[](jax.nn.get_scaled_dot_general_config.html "previous page")

previous

jax.nn.get_scaled_dot_general_config

[](jax.nn.log1mexp.html "next page")

next

jax.nn.log1mexp

Contents

- [`scaled_dot_general()`](#jax.nn.scaled_dot_general)

By The JAX authors

© Copyright 2024, The JAX Authors.\
