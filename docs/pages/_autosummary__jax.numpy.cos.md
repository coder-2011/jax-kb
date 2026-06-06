- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.cos

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.cos.rst "Download source file")
-  .pdf

# jax.numpy.cos

## Contents

- [`cos()`](#jax.numpy.cos)

# jax.numpy.cos[\#](#jax-numpy-cos "Link to this heading")

jax.numpy.cos(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L618-L649)[\#](#jax.numpy.cos "Link to this definition")  
Compute a trigonometric cosine of each element of input.

JAX implementation of [`numpy.cos`](https://numpy.org/doc/stable/reference/generated/numpy.cos.html#numpy.cos "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – scalar or array. Angle in radians.

Returns:  
An array containing the cosine of each element in `x`, promotes to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.sin()`](jax.numpy.sin.html#jax.numpy.sin "jax.numpy.sin"): Computes a trigonometric sine of each element of input.

- [`jax.numpy.tan()`](jax.numpy.tan.html#jax.numpy.tan "jax.numpy.tan"): Computes a trigonometric tangent of each element of input.

- [`jax.numpy.arccos()`](jax.numpy.arccos.html#jax.numpy.arccos "jax.numpy.arccos") and [`jax.numpy.acos()`](jax.numpy.acos.html#jax.numpy.acos "jax.numpy.acos"): Computes the inverse of trigonometric cosine of each element of input.

Examples

    >>> pi = jnp.pi
    >>> x = jnp.array([pi/4, pi/2, 3*pi/4, 5*pi/6])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   print(jnp.cos(x))
    [ 0.707 -0.    -0.707 -0.866]

[](jax.numpy.correlate.html "previous page")

previous

jax.numpy.correlate

[](jax.numpy.cosh.html "next page")

next

jax.numpy.cosh

Contents

- [`cos()`](#jax.numpy.cos)

By The JAX authors

© Copyright 2024, The JAX Authors.\
