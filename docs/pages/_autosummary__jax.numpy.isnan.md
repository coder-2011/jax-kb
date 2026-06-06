- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.isnan

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.isnan.rst "Download source file")
-  .pdf

# jax.numpy.isnan

## Contents

- [`isnan()`](#jax.numpy.isnan)

# jax.numpy.isnan[\#](#jax-numpy-isnan "Link to this heading")

jax.numpy.isnan(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3627-L3660)[\#](#jax.numpy.isnan "Link to this definition")  
Returns a boolean array indicating whether each element of input is `NaN`.

JAX implementation of [`numpy.isnan`](https://numpy.org/doc/stable/reference/generated/numpy.isnan.html#numpy.isnan "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
A boolean array of same shape as `x` containing `True` where `x` is not a number (i.e. `NaN`) and `False` otherwise.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.isfinite()`](jax.numpy.isfinite.html#jax.numpy.isfinite "jax.numpy.isfinite"): Returns a boolean array indicating whether each element of input is finite.

- [`jax.numpy.isinf()`](jax.numpy.isinf.html#jax.numpy.isinf "jax.numpy.isinf"): Returns a boolean array indicating whether each element of input is either positive or negative infinity.

- [`jax.numpy.isposinf()`](jax.numpy.isposinf.html#jax.numpy.isposinf "jax.numpy.isposinf"): Returns a boolean array indicating whether each element of input is positive infinity.

- [`jax.numpy.isneginf()`](jax.numpy.isneginf.html#jax.numpy.isneginf "jax.numpy.isneginf"): Returns a boolean array indicating whether each element of input is negative infinity.

Examples

    >>> jnp.isnan(6)
    Array(False, dtype=bool, weak_type=True)
    >>> x = jnp.array([2, 1+4j, jnp.inf, jnp.nan])
    >>> jnp.isnan(x)
    Array([False, False, False,  True], dtype=bool)

[](jax.numpy.isinf.html "previous page")

previous

jax.numpy.isinf

[](jax.numpy.isneginf.html "next page")

next

jax.numpy.isneginf

Contents

- [`isnan()`](#jax.numpy.isnan)

By The JAX authors

© Copyright 2024, The JAX Authors.\
