- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.log10

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.log10.rst "Download source file")
-  .pdf

# jax.numpy.log10

## Contents

- [`log10()`](#jax.numpy.log10)

# jax.numpy.log10[\#](#jax-numpy-log10 "Link to this heading")

jax.numpy.log10(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L2825-L2856)[\#](#jax.numpy.log10 "Link to this definition")  
Calculates the base-10 logarithm of x element-wise

JAX implementation of [`numpy.log10`](https://numpy.org/doc/stable/reference/generated/numpy.log10.html#numpy.log10 "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – Input array

Returns:  
An array containing the base-10 logarithm of each element in `x`, promotes to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> x1 = jnp.array([0.01, 0.1, 1, 10, 100, 1000])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.log10(x1))
    [-2. -1.  0.  1.  2.  3.]

[](jax.numpy.log.html "previous page")

previous

jax.numpy.log

[](jax.numpy.log1p.html "next page")

next

jax.numpy.log1p

Contents

- [`log10()`](#jax.numpy.log10)

By The JAX authors

© Copyright 2024, The JAX Authors.\
