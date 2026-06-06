- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.log2

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.log2.rst "Download source file")
-  .pdf

# jax.numpy.log2

## Contents

- [`log2()`](#jax.numpy.log2)

# jax.numpy.log2[\#](#jax-numpy-log2 "Link to this heading")

jax.numpy.log2(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L2794-L2823)[\#](#jax.numpy.log2 "Link to this definition")  
Calculates the base-2 logarithm of `x` element-wise.

JAX implementation of [`numpy.log2`](https://numpy.org/doc/stable/reference/generated/numpy.log2.html#numpy.log2 "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – Input array

Returns:  
An array containing the base-2 logarithm of each element in `x`, promotes to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> x1 = jnp.array([0.25, 0.5, 1, 2, 4, 8])
    >>> jnp.log2(x1)
    Array([-2., -1.,  0.,  1.,  2.,  3.], dtype=float32)

[](jax.numpy.log1p.html "previous page")

previous

jax.numpy.log1p

[](jax.numpy.logaddexp.html "next page")

next

jax.numpy.logaddexp

Contents

- [`log2()`](#jax.numpy.log2)

By The JAX authors

© Copyright 2024, The JAX Authors.\
