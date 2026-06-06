- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.reciprocal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.reciprocal.rst "Download source file")
-  .pdf

# jax.numpy.reciprocal

## Contents

- [`reciprocal()`](#jax.numpy.reciprocal)

# jax.numpy.reciprocal[\#](#jax-numpy-reciprocal "Link to this heading")

jax.numpy.reciprocal(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3762-L3794)[\#](#jax.numpy.reciprocal "Link to this definition")  
Calculate element-wise reciprocal of the input.

JAX implementation of [`numpy.reciprocal`](https://numpy.org/doc/stable/reference/generated/numpy.reciprocal.html#numpy.reciprocal "(in NumPy v2.4)").

The reciprocal is calculated by `1/x`.

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
An array of same shape as `x` containing the reciprocal of each element of `x`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

For integer inputs, `np.reciprocal` returns rounded integer output, while `jnp.reciprocal` promotes integer inputs to floating point.

Examples

    >>> jnp.reciprocal(2)
    Array(0.5, dtype=float32, weak_type=True)
    >>> jnp.reciprocal(0.)
    Array(inf, dtype=float32, weak_type=True)
    >>> x = jnp.array([1, 5., 4.])
    >>> jnp.reciprocal(x)
    Array([1.  , 0.2 , 0.25], dtype=float32)

[](jax.numpy.real.html "previous page")

previous

jax.numpy.real

[](jax.numpy.remainder.html "next page")

next

jax.numpy.remainder

Contents

- [`reciprocal()`](#jax.numpy.reciprocal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
