- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.arctanh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.arctanh.rst "Download source file")
-  .pdf

# jax.numpy.arctanh

## Contents

- [`arctanh()`](#jax.numpy.arctanh)

# jax.numpy.arctanh[\#](#jax-numpy-arctanh "Link to this heading")

jax.numpy.arctanh(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L1084-L1133)[\#](#jax.numpy.arctanh "Link to this definition")  
Calculate element-wise inverse of hyperbolic tangent of input.

JAX implementation of [`numpy.arctanh`](https://numpy.org/doc/stable/reference/generated/numpy.arctanh.html#numpy.arctanh "(in NumPy v2.4)").

The inverse of hyperbolic tangent is defined by:

\\arctanh(x) = \frac{1}{2} \[\ln(1 + x) - \ln(1 - x)\]\\

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
An array of same shape as `x` containing the inverse of hyperbolic tangent of each element of `x`, promoting to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

- `jnp.arctanh` returns `nan` for real-values outside the range `[-1,`` ``1]`.

- `jnp.arctanh` follows the branch cut convention of [`numpy.arctanh`](https://numpy.org/doc/stable/reference/generated/numpy.arctanh.html#numpy.arctanh "(in NumPy v2.4)") for complex inputs.

See also

- [`jax.numpy.tanh()`](jax.numpy.tanh.html#jax.numpy.tanh "jax.numpy.tanh"): Computes the element-wise hyperbolic tangent of the input.

- [`jax.numpy.arcsinh()`](jax.numpy.arcsinh.html#jax.numpy.arcsinh "jax.numpy.arcsinh"): Computes the element-wise inverse of hyperbolic sine of the input.

- [`jax.numpy.arccosh()`](jax.numpy.arccosh.html#jax.numpy.arccosh "jax.numpy.arccosh"): Computes the element-wise inverse of hyperbolic cosine of the input.

Examples

    >>> x = jnp.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arctanh(x)
    Array([   nan,   -inf, -0.549,  0.   ,  0.549,    inf,    nan], dtype=float32)

For complex-valued input:

    >>> x1 = jnp.array([-2+0j, 3+0j, 4-1j])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arctanh(x1)
    Array([-0.549+1.571j,  0.347+1.571j,  0.239-1.509j], dtype=complex64)

[](jax.numpy.arctan2.html "previous page")

previous

jax.numpy.arctan2

[](jax.numpy.argmax.html "next page")

next

jax.numpy.argmax

Contents

- [`arctanh()`](#jax.numpy.arctanh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
