- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.isfinite

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.isfinite.rst "Download source file")
-  .pdf

# jax.numpy.isfinite

## Contents

- [`isfinite()`](#jax.numpy.isfinite)

# jax.numpy.isfinite[\#](#jax-numpy-isfinite "Link to this heading")

jax.numpy.isfinite(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3460-L3499)[\#](#jax.numpy.isfinite "Link to this definition")  
Return a boolean array indicating whether each element of input is finite.

JAX implementation of [`numpy.isfinite`](https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html#numpy.isfinite "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
A boolean array of same shape as `x` containing `True` where `x` is not `inf`, `-inf`, or `NaN`, and `False` otherwise.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.isinf()`](jax.numpy.isinf.html#jax.numpy.isinf "jax.numpy.isinf"): Returns a boolean array indicating whether each element of input is either positive or negative infinity.

- [`jax.numpy.isposinf()`](jax.numpy.isposinf.html#jax.numpy.isposinf "jax.numpy.isposinf"): Returns a boolean array indicating whether each element of input is positive infinity.

- [`jax.numpy.isneginf()`](jax.numpy.isneginf.html#jax.numpy.isneginf "jax.numpy.isneginf"): Returns a boolean array indicating whether each element of input is negative infinity.

- [`jax.numpy.isnan()`](jax.numpy.isnan.html#jax.numpy.isnan "jax.numpy.isnan"): Returns a boolean array indicating whether each element of input is not a number (`NaN`).

Examples

    >>> x = jnp.array([-1, 3, jnp.inf, jnp.nan])
    >>> jnp.isfinite(x)
    Array([ True,  True, False, False], dtype=bool)
    >>> jnp.isfinite(3-4j)
    Array(True, dtype=bool, weak_type=True)

[](jax.numpy.isdtype.html "previous page")

previous

jax.numpy.isdtype

[](jax.numpy.isin.html "next page")

next

jax.numpy.isin

Contents

- [`isfinite()`](#jax.numpy.isfinite)

By The JAX authors

© Copyright 2024, The JAX Authors.\
