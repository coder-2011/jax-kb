- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.arcsin

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.arcsin.rst "Download source file")
-  .pdf

# jax.numpy.arcsin

## Contents

- [`arcsin()`](#jax.numpy.arcsin)

# jax.numpy.arcsin[\#](#jax-numpy-arcsin "Link to this heading")

jax.numpy.arcsin(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L684-L726)[\#](#jax.numpy.arcsin "Link to this definition")  
Compute element-wise inverse of trigonometric sine of input.

JAX implementation of [`numpy.arcsin`](https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html#numpy.arcsin "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
An array containing the inverse trigonometric sine of each element of `x` in radians in the range `[-pi/2,`` ``pi/2]`, promoting to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

- `jnp.arcsin` returns `nan` when `x` is real-valued and not in the closed interval `[-1,`` ``1]`.

- `jnp.arcsin` follows the branch cut convention of [`numpy.arcsin`](https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html#numpy.arcsin "(in NumPy v2.4)") for complex inputs.

See also

- [`jax.numpy.sin()`](jax.numpy.sin.html#jax.numpy.sin "jax.numpy.sin"): Computes a trigonometric sine of each element of input.

- [`jax.numpy.arccos()`](jax.numpy.arccos.html#jax.numpy.arccos "jax.numpy.arccos") and [`jax.numpy.acos()`](jax.numpy.acos.html#jax.numpy.acos "jax.numpy.acos"): Computes the inverse of trigonometric cosine of each element of input.

- [`jax.numpy.arctan()`](jax.numpy.arctan.html#jax.numpy.arctan "jax.numpy.arctan") and [`jax.numpy.atan()`](jax.numpy.atan.html#jax.numpy.atan "jax.numpy.atan"): Computes the inverse of trigonometric tangent of each element of input.

Examples

    >>> x = jnp.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arcsin(x)
    Array([   nan, -1.571, -0.524,  0.   ,  0.524,  1.571,    nan], dtype=float32)

For complex-valued inputs:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arcsin(3+4j)
    Array(0.634+2.306j, dtype=complex64, weak_type=True)

[](jax.numpy.arccosh.html "previous page")

previous

jax.numpy.arccosh

[](jax.numpy.arcsinh.html "next page")

next

jax.numpy.arcsinh

Contents

- [`arcsin()`](#jax.numpy.arcsin)

By The JAX authors

© Copyright 2024, The JAX Authors.\
