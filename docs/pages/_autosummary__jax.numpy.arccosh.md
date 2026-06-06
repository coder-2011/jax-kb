- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.arccosh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.arccosh.rst "Download source file")
-  .pdf

# jax.numpy.arccosh

## Contents

- [`arccosh()`](#jax.numpy.arccosh)

# jax.numpy.arccosh[\#](#jax-numpy-arccosh "Link to this heading")

jax.numpy.arccosh(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L973-L1028)[\#](#jax.numpy.arccosh "Link to this definition")  
Calculate element-wise inverse of hyperbolic cosine of input.

JAX implementation of [`numpy.arccosh`](https://numpy.org/doc/stable/reference/generated/numpy.arccosh.html#numpy.arccosh "(in NumPy v2.4)").

The inverse of hyperbolic cosine is defined by:

\\arccosh(x) = \ln(x + \sqrt{x^2 - 1})\\

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
An array of same shape as `x` containing the inverse of hyperbolic cosine of each element of `x`, promoting to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

- `jnp.arccosh` returns `nan` for real-values in the range `[-inf,`` ``1)`.

- `jnp.arccosh` follows the branch cut convention of [`numpy.arccosh`](https://numpy.org/doc/stable/reference/generated/numpy.arccosh.html#numpy.arccosh "(in NumPy v2.4)") for complex inputs.

See also

- [`jax.numpy.cosh()`](jax.numpy.cosh.html#jax.numpy.cosh "jax.numpy.cosh"): Computes the element-wise hyperbolic cosine of the input.

- [`jax.numpy.arcsinh()`](jax.numpy.arcsinh.html#jax.numpy.arcsinh "jax.numpy.arcsinh"): Computes the element-wise inverse of hyperbolic sine of the input.

- [`jax.numpy.arctanh()`](jax.numpy.arctanh.html#jax.numpy.arctanh "jax.numpy.arctanh"): Computes the element-wise inverse of hyperbolic tangent of the input.

Examples

    >>> x = jnp.array([[1, 3, -4],
    ...                [-5, 2, 7]])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arccosh(x)
    Array([[0.   , 1.763,   nan],
           [  nan, 1.317, 2.634]], dtype=float32)

For complex-valued input:

    >>> x1 = jnp.array([-jnp.inf+0j, 1+2j, -5+0j])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arccosh(x1)
    Array([  inf+3.142j, 1.529+1.144j, 2.292+3.142j], dtype=complex64)

[](jax.numpy.arccos.html "previous page")

previous

jax.numpy.arccos

[](jax.numpy.arcsin.html "next page")

next

jax.numpy.arcsin

Contents

- [`arccosh()`](#jax.numpy.arccosh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
