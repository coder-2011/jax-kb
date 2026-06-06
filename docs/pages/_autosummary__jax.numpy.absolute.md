- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.absolute

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.absolute.rst "Download source file")
-  .pdf

# jax.numpy.absolute

## Contents

- [`absolute()`](#jax.numpy.absolute)

# jax.numpy.absolute[\#](#jax-numpy-absolute "Link to this heading")

jax.numpy.absolute(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L2314-L2352)[\#](#jax.numpy.absolute "Link to this definition")  
Calculate the absolute value element-wise.

JAX implementation of [`numpy.absolute`](https://numpy.org/doc/stable/reference/generated/numpy.absolute.html#numpy.absolute "(in NumPy v2.4)").

This is the same function as [`jax.numpy.abs()`](jax.numpy.abs.html#jax.numpy.abs "jax.numpy.abs").

Parameters:  
**x** (*ArrayLike*) – Input array

Returns:  
An array-like object containing the absolute value of each element in `x`, with the same shape as `x`. For complex valued input, \\a + ib\\, the absolute value is \\\sqrt{a^2+b^2}\\.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

For complex inputs containing `NaN` in the real or imaginary part, `abs` will always return `NaN`.

Examples

    >>> x1 = jnp.array([5, -2, 0, 12])
    >>> jnp.absolute(x1)
    Array([ 5,  2,  0, 12], dtype=int32)

    >>> x2 = jnp.array([[ 8, -3, 1],[ 0, 9, -6]])
    >>> jnp.absolute(x2)
    Array([[8, 3, 1],
           [0, 9, 6]], dtype=int32)

    >>> x3 = jnp.array([8 + 15j, 3 - 4j, -5 + 0j])
    >>> jnp.absolute(x3)
    Array([17.,  5.,  5.], dtype=float32)

[](jax.numpy.abs.html "previous page")

previous

jax.numpy.abs

[](jax.numpy.acos.html "next page")

next

jax.numpy.acos

Contents

- [`absolute()`](#jax.numpy.absolute)

By The JAX authors

© Copyright 2024, The JAX Authors.\
