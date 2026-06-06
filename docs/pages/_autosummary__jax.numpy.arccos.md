- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.arccos

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.arccos.rst "Download source file")
-  .pdf

# jax.numpy.arccos

## Contents

- [`arccos()`](#jax.numpy.arccos)

# jax.numpy.arccos[\#](#jax-numpy-arccos "Link to this heading")

jax.numpy.arccos(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L728-L771)[\#](#jax.numpy.arccos "Link to this definition")  
Compute element-wise inverse of trigonometric cosine of input.

JAX implementation of [`numpy.arccos`](https://numpy.org/doc/stable/reference/generated/numpy.arccos.html#numpy.arccos "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
An array containing the inverse trigonometric cosine of each element of `x` in radians in the range `[0,`` ``pi]`, promoting to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

- `jnp.arccos` returns `nan` when `x` is real-valued and not in the closed interval `[-1,`` ``1]`.

- `jnp.arccos` follows the branch cut convention of [`numpy.arccos`](https://numpy.org/doc/stable/reference/generated/numpy.arccos.html#numpy.arccos "(in NumPy v2.4)") for complex inputs.

See also

- [`jax.numpy.cos()`](jax.numpy.cos.html#jax.numpy.cos "jax.numpy.cos"): Computes a trigonometric cosine of each element of input.

- [`jax.numpy.arcsin()`](jax.numpy.arcsin.html#jax.numpy.arcsin "jax.numpy.arcsin") and [`jax.numpy.asin()`](jax.numpy.asin.html#jax.numpy.asin "jax.numpy.asin"): Computes the inverse of trigonometric sine of each element of input.

- [`jax.numpy.arctan()`](jax.numpy.arctan.html#jax.numpy.arctan "jax.numpy.arctan") and [`jax.numpy.atan()`](jax.numpy.atan.html#jax.numpy.atan "jax.numpy.atan"): Computes the inverse of trigonometric tangent of each element of input.

Examples

    >>> x = jnp.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arccos(x)
    Array([  nan, 3.142, 2.094, 1.571, 1.047, 0.   ,   nan], dtype=float32)

For complex inputs:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arccos(4-1j)
    Array(0.252+2.097j, dtype=complex64, weak_type=True)

[](jax.numpy.arange.html "previous page")

previous

jax.numpy.arange

[](jax.numpy.arccosh.html "next page")

next

jax.numpy.arccosh

Contents

- [`arccos()`](#jax.numpy.arccos)

By The JAX authors

© Copyright 2024, The JAX Authors.\
