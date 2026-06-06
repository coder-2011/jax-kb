- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.tanh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.tanh.rst "Download source file")
-  .pdf

# jax.numpy.tanh

## Contents

- [`tanh()`](#jax.numpy.tanh)

# jax.numpy.tanh[\#](#jax-numpy-tanh "Link to this heading")

jax.numpy.tanh(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L1030-L1082)[\#](#jax.numpy.tanh "Link to this definition")  
Calculate element-wise hyperbolic tangent of input.

JAX implementation of [`numpy.tanh`](https://numpy.org/doc/stable/reference/generated/numpy.tanh.html#numpy.tanh "(in NumPy v2.4)").

The hyperbolic tangent is defined by:

\\tanh(x) = \frac{sinh(x)}{cosh(x)} = \frac{e^x - e^{-x}}{e^x + e^{-x}}\\

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
An array containing the hyperbolic tangent of each element of `x`, promoting to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

`jnp.tanh` is equivalent to computing `-1j`` ``*`` ``jnp.tan(1j`` ``*`` ``x)`.

See also

- [`jax.numpy.sinh()`](jax.numpy.sinh.html#jax.numpy.sinh "jax.numpy.sinh"): Computes the element-wise hyperbolic sine of the input.

- [`jax.numpy.cosh()`](jax.numpy.cosh.html#jax.numpy.cosh "jax.numpy.cosh"): Computes the element-wise hyperbolic cosine of the input.

- [`jax.numpy.arctanh()`](jax.numpy.arctanh.html#jax.numpy.arctanh "jax.numpy.arctanh"): Computes the element-wise inverse of hyperbolic tangent of the input.

Examples

    >>> x = jnp.array([[-1, 0, 1],
    ...                [3, -2, 5]])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.tanh(x)
    Array([[-0.762,  0.   ,  0.762],
           [ 0.995, -0.964,  1.   ]], dtype=float32)
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   -1j * jnp.tan(1j * x)
    Array([[-0.762+0.j,  0.   -0.j,  0.762-0.j],
           [ 0.995-0.j, -0.964+0.j,  1.   -0.j]],      dtype=complex64, weak_type=True)

For complex-valued input:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.tanh(2-5j)
    Array(1.031+0.021j, dtype=complex64, weak_type=True)
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   -1j * jnp.tan(1j * (2-5j))
    Array(1.031+0.021j, dtype=complex64, weak_type=True)

[](jax.numpy.tan.html "previous page")

previous

jax.numpy.tan

[](jax.numpy.tensordot.html "next page")

next

jax.numpy.tensordot

Contents

- [`tanh()`](#jax.numpy.tanh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
