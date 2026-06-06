- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.arctan

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.arctan.rst "Download source file")
-  .pdf

# jax.numpy.arctan

## Contents

- [`arctan()`](#jax.numpy.arctan)

# jax.numpy.arctan[\#](#jax-numpy-arctan "Link to this heading")

jax.numpy.arctan(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L773-L812)[\#](#jax.numpy.arctan "Link to this definition")  
Compute element-wise inverse of trigonometric tangent of input.

JAX implement of [`numpy.arctan`](https://numpy.org/doc/stable/reference/generated/numpy.arctan.html#numpy.arctan "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
An array containing the inverse trigonometric tangent of each element `x` in radians in the range `[-pi/2,`` ``pi/2]`, promoting to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

`jnp.arctan` follows the branch cut convention of [`numpy.arctan`](https://numpy.org/doc/stable/reference/generated/numpy.arctan.html#numpy.arctan "(in NumPy v2.4)") for complex inputs.

See also

- [`jax.numpy.tan()`](jax.numpy.tan.html#jax.numpy.tan "jax.numpy.tan"): Computes a trigonometric tangent of each element of input.

- [`jax.numpy.arcsin()`](jax.numpy.arcsin.html#jax.numpy.arcsin "jax.numpy.arcsin") and [`jax.numpy.asin()`](jax.numpy.asin.html#jax.numpy.asin "jax.numpy.asin"): Computes the inverse of trigonometric sine of each element of input.

- [`jax.numpy.arccos()`](jax.numpy.arccos.html#jax.numpy.arccos "jax.numpy.arccos") and [`jax.numpy.atan()`](jax.numpy.atan.html#jax.numpy.atan "jax.numpy.atan"): Computes the inverse of trigonometric cosine of each element of input.

Examples

    >>> x = jnp.array([-jnp.inf, -20, -1, 0, 1, 20, jnp.inf])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arctan(x)
    Array([-1.571, -1.521, -0.785,  0.   ,  0.785,  1.521,  1.571], dtype=float32)

For complex-valued inputs:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arctan(2+7j)
    Array(1.532+0.133j, dtype=complex64, weak_type=True)

[](jax.numpy.arcsinh.html "previous page")

previous

jax.numpy.arcsinh

[](jax.numpy.arctan2.html "next page")

next

jax.numpy.arctan2

Contents

- [`arctan()`](#jax.numpy.arctan)

By The JAX authors

© Copyright 2024, The JAX Authors.\
