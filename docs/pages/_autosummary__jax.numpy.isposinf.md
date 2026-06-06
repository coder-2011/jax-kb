- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.isposinf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.isposinf.rst "Download source file")
-  .pdf

# jax.numpy.isposinf

## Contents

- [`isposinf()`](#jax.numpy.isposinf)

# jax.numpy.isposinf[\#](#jax-numpy-isposinf "Link to this heading")

jax.numpy.isposinf(*x*, */*, *out=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3557-L3590)[\#](#jax.numpy.isposinf "Link to this definition")  
Return boolean array indicating whether each element of input is positive infinite.

JAX implementation of [`numpy.isposinf`](https://numpy.org/doc/stable/reference/generated/numpy.isposinf.html#numpy.isposinf "(in NumPy v2.4)").

Parameters:  
**x** – input array or scalar. `complex` dtype are not supported.

Returns:  
A boolean array of same shape as `x` containing `True` where `x` is `inf`, and `False` otherwise.

See also

- [`jax.numpy.isinf()`](jax.numpy.isinf.html#jax.numpy.isinf "jax.numpy.isinf"): Returns a boolean array indicating whether each element of input is either positive or negative infinity.

- [`jax.numpy.isneginf()`](jax.numpy.isneginf.html#jax.numpy.isneginf "jax.numpy.isneginf"): Returns a boolean array indicating whether each element of input is negative infinity.

- [`jax.numpy.isfinite()`](jax.numpy.isfinite.html#jax.numpy.isfinite "jax.numpy.isfinite"): Returns a boolean array indicating whether each element of input is finite.

- [`jax.numpy.isnan()`](jax.numpy.isnan.html#jax.numpy.isnan "jax.numpy.isnan"): Returns a boolean array indicating whether each element of input is not a number (`NaN`).

Examples

    >>> jnp.isposinf(5)
    Array(False, dtype=bool)
    >>> x = jnp.array([-jnp.inf, 5, jnp.inf, jnp.nan, 1])
    >>> jnp.isposinf(x)
    Array([False, False,  True, False, False], dtype=bool)

[](jax.numpy.isneginf.html "previous page")

previous

jax.numpy.isneginf

[](jax.numpy.isreal.html "next page")

next

jax.numpy.isreal

Contents

- [`isposinf()`](#jax.numpy.isposinf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
