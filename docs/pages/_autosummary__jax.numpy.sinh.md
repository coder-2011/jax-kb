- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.sinh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.sinh.rst "Download source file")
-  .pdf

# jax.numpy.sinh

## Contents

- [`sinh()`](#jax.numpy.sinh)

# jax.numpy.sinh[\#](#jax-numpy-sinh "Link to this heading")

jax.numpy.sinh(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L814-L867)[\#](#jax.numpy.sinh "Link to this definition")  
Calculate element-wise hyperbolic sine of input.

JAX implementation of [`numpy.sinh`](https://numpy.org/doc/stable/reference/generated/numpy.sinh.html#numpy.sinh "(in NumPy v2.4)").

The hyperbolic sine is defined by:

\\sinh(x) = \frac{e^x - e^{-x}}{2}\\

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
An array containing the hyperbolic sine of each element of `x`, promoting to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

`jnp.sinh` is equivalent to computing `-1j`` ``*`` ``jnp.sin(1j`` ``*`` ``x)`.

See also

- [`jax.numpy.cosh()`](jax.numpy.cosh.html#jax.numpy.cosh "jax.numpy.cosh"): Computes the element-wise hyperbolic cosine of the input.

- [`jax.numpy.tanh()`](jax.numpy.tanh.html#jax.numpy.tanh "jax.numpy.tanh"): Computes the element-wise hyperbolic tangent of the input.

- [`jax.numpy.arcsinh()`](jax.numpy.arcsinh.html#jax.numpy.arcsinh "jax.numpy.arcsinh"): Computes the element-wise inverse of hyperbolic sine of the input.

Examples

    >>> x = jnp.array([[-2, 3, 5],
    ...                [0, -1, 4]])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.sinh(x)
    Array([[-3.627, 10.018, 74.203],
           [ 0.   , -1.175, 27.29 ]], dtype=float32)
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   -1j * jnp.sin(1j * x)
    Array([[-3.627+0.j, 10.018-0.j, 74.203-0.j],
           [ 0.   -0.j, -1.175+0.j, 27.29 -0.j]],      dtype=complex64, weak_type=True)

For complex-valued input:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.sinh(3-2j)
    Array(-4.169-9.154j, dtype=complex64, weak_type=True)
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   -1j * jnp.sin(1j * (3-2j))
    Array(-4.169-9.154j, dtype=complex64, weak_type=True)

[](jax.numpy.single.html "previous page")

previous

jax.numpy.single

[](jax.numpy.size.html "next page")

next

jax.numpy.size

Contents

- [`sinh()`](#jax.numpy.sinh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
