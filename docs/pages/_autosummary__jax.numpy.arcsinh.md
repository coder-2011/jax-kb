- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.arcsinh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.arcsinh.rst "Download source file")
-  .pdf

# jax.numpy.arcsinh

## Contents

- [`arcsinh()`](#jax.numpy.arcsinh)

# jax.numpy.arcsinh[\#](#jax-numpy-arcsinh "Link to this heading")

jax.numpy.arcsinh(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L923-L971)[\#](#jax.numpy.arcsinh "Link to this definition")  
Calculate element-wise inverse of hyperbolic sine of input.

JAX implementation of [`numpy.arcsinh`](https://numpy.org/doc/stable/reference/generated/numpy.arcsinh.html#numpy.arcsinh "(in NumPy v2.4)").

The inverse of hyperbolic sine is defined by:

\\arcsinh(x) = \ln(x + \sqrt{1 + x^2})\\

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
An array of same shape as `x` containing the inverse of hyperbolic sine of each element of `x`, promoting to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

- `jnp.arcsinh` returns `nan` for values outside the range `(-inf,`` ``inf)`.

- `jnp.arcsinh` follows the branch cut convention of [`numpy.arcsinh`](https://numpy.org/doc/stable/reference/generated/numpy.arcsinh.html#numpy.arcsinh "(in NumPy v2.4)") for complex inputs.

See also

- [`jax.numpy.sinh()`](jax.numpy.sinh.html#jax.numpy.sinh "jax.numpy.sinh"): Computes the element-wise hyperbolic sine of the input.

- [`jax.numpy.arccosh()`](jax.numpy.arccosh.html#jax.numpy.arccosh "jax.numpy.arccosh"): Computes the element-wise inverse of hyperbolic cosine of the input.

- [`jax.numpy.arctanh()`](jax.numpy.arctanh.html#jax.numpy.arctanh "jax.numpy.arctanh"): Computes the element-wise inverse of hyperbolic tangent of the input.

Examples

    >>> x = jnp.array([[-2, 3, 1],
    ...                [4, 9, -5]])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arcsinh(x)
    Array([[-1.444,  1.818,  0.881],
           [ 2.095,  2.893, -2.312]], dtype=float32)

For complex-valued inputs:

    >>> x1 = jnp.array([4-3j, 2j])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arcsinh(x1)
    Array([2.306-0.634j, 1.317+1.571j], dtype=complex64)

[](jax.numpy.arcsin.html "previous page")

previous

jax.numpy.arcsin

[](jax.numpy.arctan.html "next page")

next

jax.numpy.arctan

Contents

- [`arcsinh()`](#jax.numpy.arcsinh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
