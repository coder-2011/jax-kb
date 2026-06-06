- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.tan

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.tan.rst "Download source file")
-  .pdf

# jax.numpy.tan

## Contents

- [`tan()`](#jax.numpy.tan)

# jax.numpy.tan[\#](#jax-numpy-tan "Link to this heading")

jax.numpy.tan(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L651-L682)[\#](#jax.numpy.tan "Link to this definition")  
Compute a trigonometric tangent of each element of input.

JAX implementation of [`numpy.tan`](https://numpy.org/doc/stable/reference/generated/numpy.tan.html#numpy.tan "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – scalar or array. Angle in radians.

Returns:  
An array containing the tangent of each element in `x`, promotes to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.sin()`](jax.numpy.sin.html#jax.numpy.sin "jax.numpy.sin"): Computes a trigonometric sine of each element of input.

- [`jax.numpy.cos()`](jax.numpy.cos.html#jax.numpy.cos "jax.numpy.cos"): Computes a trigonometric cosine of each element of input.

- [`jax.numpy.arctan()`](jax.numpy.arctan.html#jax.numpy.arctan "jax.numpy.arctan") and [`jax.numpy.atan()`](jax.numpy.atan.html#jax.numpy.atan "jax.numpy.atan"): Computes the inverse of trigonometric tangent of each element of input.

Examples

    >>> pi = jnp.pi
    >>> x = jnp.array([0, pi/6, pi/4, 3*pi/4, 5*pi/6])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   print(jnp.tan(x))
    [ 0.     0.577  1.    -1.    -0.577]

[](jax.numpy.take_along_axis.html "previous page")

previous

jax.numpy.take_along_axis

[](jax.numpy.tanh.html "next page")

next

jax.numpy.tanh

Contents

- [`tan()`](#jax.numpy.tan)

By The JAX authors

© Copyright 2024, The JAX Authors.\
