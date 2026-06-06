- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.isinf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.isinf.rst "Download source file")
-  .pdf

# jax.numpy.isinf

## Contents

- [`isinf()`](#jax.numpy.isinf)

# jax.numpy.isinf[\#](#jax-numpy-isinf "Link to this heading")

jax.numpy.isinf(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3501-L3543)[\#](#jax.numpy.isinf "Link to this definition")  
Return a boolean array indicating whether each element of input is infinite.

JAX implementation of [`numpy.isinf`](https://numpy.org/doc/stable/reference/generated/numpy.isinf.html#numpy.isinf "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
A boolean array of same shape as `x` containing `True` where `x` is `inf` or `-inf`, and `False` otherwise.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.isposinf()`](jax.numpy.isposinf.html#jax.numpy.isposinf "jax.numpy.isposinf"): Returns a boolean array indicating whether each element of input is positive infinity.

- [`jax.numpy.isneginf()`](jax.numpy.isneginf.html#jax.numpy.isneginf "jax.numpy.isneginf"): Returns a boolean array indicating whether each element of input is negative infinity.

- [`jax.numpy.isfinite()`](jax.numpy.isfinite.html#jax.numpy.isfinite "jax.numpy.isfinite"): Returns a boolean array indicating whether each element of input is finite.

- [`jax.numpy.isnan()`](jax.numpy.isnan.html#jax.numpy.isnan "jax.numpy.isnan"): Returns a boolean array indicating whether each element of input is not a number (`NaN`).

Examples

    >>> jnp.isinf(jnp.inf)
    Array(True, dtype=bool)
    >>> x = jnp.array([2+3j, -jnp.inf, 6, jnp.inf, jnp.nan])
    >>> jnp.isinf(x)
    Array([False,  True, False,  True, False], dtype=bool)

[](jax.numpy.isin.html "previous page")

previous

jax.numpy.isin

[](jax.numpy.isnan.html "next page")

next

jax.numpy.isnan

Contents

- [`isinf()`](#jax.numpy.isinf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
