- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.sin

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.sin.rst "Download source file")
-  .pdf

# jax.numpy.sin

## Contents

- [`sin()`](#jax.numpy.sin)

# jax.numpy.sin[\#](#jax-numpy-sin "Link to this heading")

jax.numpy.sin(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L584-L616)[\#](#jax.numpy.sin "Link to this definition")  
Compute a trigonometric sine of each element of input.

JAX implementation of [`numpy.sin`](https://numpy.org/doc/stable/reference/generated/numpy.sin.html#numpy.sin "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – array or scalar. Angle in radians.

Returns:  
An array containing the sine of each element in `x`, promotes to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.cos()`](jax.numpy.cos.html#jax.numpy.cos "jax.numpy.cos"): Computes a trigonometric cosine of each element of input.

- [`jax.numpy.tan()`](jax.numpy.tan.html#jax.numpy.tan "jax.numpy.tan"): Computes a trigonometric tangent of each element of input.

- [`jax.numpy.arcsin()`](jax.numpy.arcsin.html#jax.numpy.arcsin "jax.numpy.arcsin") and [`jax.numpy.asin()`](jax.numpy.asin.html#jax.numpy.asin "jax.numpy.asin"): Computes the inverse of trigonometric sine of each element of input.

Examples

    >>> pi = jnp.pi
    >>> x = jnp.array([pi/4, pi/2, 3*pi/4, pi])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   print(jnp.sin(x))
    [ 0.707  1.     0.707 -0.   ]

[](jax.numpy.signedinteger.html "previous page")

previous

jax.numpy.signedinteger

[](jax.numpy.sinc.html "next page")

next

jax.numpy.sinc

Contents

- [`sin()`](#jax.numpy.sin)

By The JAX authors

© Copyright 2024, The JAX Authors.\
