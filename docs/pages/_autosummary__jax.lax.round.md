- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.round

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.round.rst "Download source file")
-  .pdf

# jax.lax.round

## Contents

- [`round()`](#jax.lax.round)

# jax.lax.round[\#](#jax-lax-round "Link to this heading")

jax.lax.round(*x*, *rounding_method=RoundingMethod.AWAY_FROM_ZERO*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L394-L429)[\#](#jax.lax.round "Link to this definition")  
Elementwise round.

Rounds values to the nearest integer. This function lowers directly to the [stablehlo.round](https://openxla.org/stablehlo/spec#round) operation.

Parameters:  
- **x** (*ArrayLike*) – an array or scalar value to round. Must have floating-point type.

- **rounding_method** ([*RoundingMethod*](../jax.lax.html#jax.lax.RoundingMethod "jax.lax.RoundingMethod")) – the method to use when rounding halfway values (e.g., `0.5`). See [`jax.lax.RoundingMethod`](../jax.lax.html#jax.lax.RoundingMethod "jax.lax.RoundingMethod") for possible values.

Returns:  
An array of the same shape and dtype as `x`, containing the elementwise rounding of `x`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.floor()`](jax.lax.floor.html#jax.lax.floor "jax.lax.floor"): round to the next integer toward negative infinity

- [`jax.lax.ceil()`](jax.lax.ceil.html#jax.lax.ceil "jax.lax.ceil"): round to the next integer toward positive infinity

Examples

    >>> import jax.numpy as jnp
    >>> from jax import lax
    >>> x = jnp.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    >>> jax.lax.round(x)  # defaults method is AWAY_FROM_ZERO
    Array([-2., -1., -1.,  0.,  1.,  1.,  2.], dtype=float32)
    >>> jax.lax.round(x, rounding_method=jax.lax.RoundingMethod.TO_NEAREST_EVEN)
    Array([-2., -1., -0.,  0.,  0.,  1.,  2.], dtype=float32)

[](jax.lax.rng_uniform.html "previous page")

previous

jax.lax.rng_uniform

[](jax.lax.rsqrt.html "next page")

next

jax.lax.rsqrt

Contents

- [`round()`](#jax.lax.round)

By The JAX authors

© Copyright 2024, The JAX Authors.\
