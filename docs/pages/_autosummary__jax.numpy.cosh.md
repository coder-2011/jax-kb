- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.cosh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.cosh.rst "Download source file")
-  .pdf

# jax.numpy.cosh

## Contents

- [`cosh()`](#jax.numpy.cosh)

# jax.numpy.cosh[\#](#jax-numpy-cosh "Link to this heading")

jax.numpy.cosh(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L869-L921)[\#](#jax.numpy.cosh "Link to this definition")  
Calculate element-wise hyperbolic cosine of input.

JAX implementation of [`numpy.cosh`](https://numpy.org/doc/stable/reference/generated/numpy.cosh.html#numpy.cosh "(in NumPy v2.4)").

The hyperbolic cosine is defined by:

\\cosh(x) = \frac{e^x + e^{-x}}{2}\\

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
An array containing the hyperbolic cosine of each element of `x`, promoting to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

`jnp.cosh` is equivalent to computing `jnp.cos(1j`` ``*`` ``x)`.

See also

- [`jax.numpy.sinh()`](jax.numpy.sinh.html#jax.numpy.sinh "jax.numpy.sinh"): Computes the element-wise hyperbolic sine of the input.

- [`jax.numpy.tanh()`](jax.numpy.tanh.html#jax.numpy.tanh "jax.numpy.tanh"): Computes the element-wise hyperbolic tangent of the input.

- [`jax.numpy.arccosh()`](jax.numpy.arccosh.html#jax.numpy.arccosh "jax.numpy.arccosh"): Computes the element-wise inverse of hyperbolic cosine of the input.

Examples

    >>> x = jnp.array([[3, -1, 0],
    ...                [4, 7, -5]])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.cosh(x)
    Array([[ 10.068,   1.543,   1.   ],
           [ 27.308, 548.317,  74.21 ]], dtype=float32)
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.cos(1j * x)
    Array([[ 10.068+0.j,   1.543+0.j,   1.   +0.j],
           [ 27.308+0.j, 548.317+0.j,  74.21 +0.j]],      dtype=complex64, weak_type=True)

For complex-valued input:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.cosh(5+1j)
    Array(40.096+62.44j, dtype=complex64, weak_type=True)
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.cos(1j * (5+1j))
    Array(40.096+62.44j, dtype=complex64, weak_type=True)

[](jax.numpy.cos.html "previous page")

previous

jax.numpy.cos

[](jax.numpy.count_nonzero.html "next page")

next

jax.numpy.count_nonzero

Contents

- [`cosh()`](#jax.numpy.cosh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
