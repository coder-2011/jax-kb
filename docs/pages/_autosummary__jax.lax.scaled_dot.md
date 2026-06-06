- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.scaled_dot

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.scaled_dot.rst "Download source file")
-  .pdf

# jax.lax.scaled_dot

## Contents

- [`scaled_dot()`](#jax.lax.scaled_dot)

# jax.lax.scaled_dot[\#](#jax-lax-scaled-dot "Link to this heading")

jax.lax.scaled_dot(*lhs*, *rhs*, *\**, *lhs_scale=None*, *rhs_scale=None*, *dimension_numbers=None*, *preferred_element_type=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/scaled_dot.py#L299-L419)[\#](#jax.lax.scaled_dot "Link to this definition")  
Computes a scaled dot product.

This function computes (lhs \* lhs_scale) @ (rhs \* rhs_scale) in preferred_element_type precision, where @ denotes jax.lax.dot_general.

Non-contracting dimensions of the operand and scale must have the same size. Contracting dimension size of the operand must be an integer multiple of the scale contracting dimension size (subchannel size). Latency of the op depends on what subchannel sizes are natively supported on your platform.

Note

This currently isn’t differentiable (no transpose rule).

Example

    B = 32
    M = 16384
    N = 16
    K = 4096
    subchannel_size = 32

    lhs_shape = (B, M, K)
    rhs_shape = (B, K, N)
    lhs_scales_shape = (B, M, K // subchannel_size)
    rhs_scales_shape = (B, K // subchannel_size, N)

    key = jax.random.key(42)

    lhs = jax.random.normal(key, lhs_shape, dtype=jnp.float8_e4m3fn)
    rhs = jax.random.normal(key, rhs_shape, dtype=jnp.float8_e4m3fn)
    lhs_scales = jax.random.normal(
        key, lhs_scales_shape, dtype=jnp.float8_e8m0fnu
    )
    rhs_scales = jax.random.normal(
        key, rhs_scales_shape, dtype=jnp.float8_e8m0fnu
    )

    @jax.jit
    def scaled_dot_fn(lhs, rhs, lhs_scale, rhs_scale):
      return jax.lax.scaled_dot(
          lhs,
          rhs,
          lhs_scale=lhs_scale,
          rhs_scale=rhs_scale,
          preferred_element_type=jnp.bfloat16,
      )

    result = scaled_dot_fn(
        lhs,
        rhs,
        lhs_scale=lhs_scales,
        rhs_scale=rhs_scales,
    )

Parameters:  
- **lhs** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – The left-hand side operand of the dot product.

- **rhs** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – The right-hand side operand of the dot product.

- **lhs_scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*) – The scale factor for lhs. It should be at least 2x smaller along the contracting dimension as compared to the operand.

- **rhs_scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*) – The scale factor for rhs. It should be at least 2x smaller along the contracting dimension as compared to the operand.

- **dimension_numbers** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\],* [*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]\],* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\],* [*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]\]\]* *\|* *None*) – A tuple of tuples of the form ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims)). If not provided, default is (((1,), (0,)), ((), ())) for 2D inputs which is lhs_contracting_dim=1, rhs_contracting_dim=0, and (((2,), (1,)), ((0,), (0,))) for 3D inputs which is lhs_contracting_dim=2, rhs_contracting_dim=1 and lhs_batch_dim=0, rhs_batch_dim=0.

- **preferred_element_type** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* [*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") *\|* *SupportsDType* *\|* *None*) – The desired dtype of the output and intermediate accumulations, can be bfloat16 or float32. Defaults to bfloat16.

Returns:  
The result of the scaled dot product.

[](jax.lax.rsqrt.html "previous page")

previous

jax.lax.rsqrt

[](jax.lax.scatter.html "next page")

next

jax.lax.scatter

Contents

- [`scaled_dot()`](#jax.lax.scaled_dot)

By The JAX authors

© Copyright 2024, The JAX Authors.\
