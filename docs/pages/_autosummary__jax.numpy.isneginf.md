- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.isneginf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.isneginf.rst "Download source file")
-  .pdf

# jax.numpy.isneginf

## Contents

- [`isneginf()`](#jax.numpy.isneginf)

# jax.numpy.isneginf[\#](#jax-numpy-isneginf "Link to this heading")

jax.numpy.isneginf(*x*, */*, *out=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3592-L3625)[\#](#jax.numpy.isneginf "Link to this definition")  
Return boolean array indicating whether each element of input is negative infinite.

JAX implementation of [`numpy.isneginf`](https://numpy.org/doc/stable/reference/generated/numpy.isneginf.html#numpy.isneginf "(in NumPy v2.4)").

Parameters:  
**x** – input array or scalar. `complex` dtype are not supported.

Returns:  
A boolean array of same shape as `x` containing `True` where `x` is `-inf`, and `False` otherwise.

See also

- [`jax.numpy.isinf()`](jax.numpy.isinf.html#jax.numpy.isinf "jax.numpy.isinf"): Returns a boolean array indicating whether each element of input is either positive or negative infinity.

- [`jax.numpy.isposinf()`](jax.numpy.isposinf.html#jax.numpy.isposinf "jax.numpy.isposinf"): Returns a boolean array indicating whether each element of input is positive infinity.

- [`jax.numpy.isfinite()`](jax.numpy.isfinite.html#jax.numpy.isfinite "jax.numpy.isfinite"): Returns a boolean array indicating whether each element of input is finite.

- [`jax.numpy.isnan()`](jax.numpy.isnan.html#jax.numpy.isnan "jax.numpy.isnan"): Returns a boolean array indicating whether each element of input is not a number (`NaN`).

Examples

    >>> jnp.isneginf(jnp.inf)
    Array(False, dtype=bool)
    >>> x = jnp.array([-jnp.inf, 5, jnp.inf, jnp.nan, 1])
    >>> jnp.isneginf(x)
    Array([ True, False, False, False, False], dtype=bool)

[](jax.numpy.isnan.html "previous page")

previous

jax.numpy.isnan

[](jax.numpy.isposinf.html "next page")

next

jax.numpy.isposinf

Contents

- [`isneginf()`](#jax.numpy.isneginf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
