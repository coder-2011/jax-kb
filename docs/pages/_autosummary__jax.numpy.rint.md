- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.rint

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.rint.rst "Download source file")
-  .pdf

# jax.numpy.rint

## Contents

- [`rint()`](#jax.numpy.rint)

# jax.numpy.rint[\#](#jax-numpy-rint "Link to this heading")

jax.numpy.rint(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L2361-L2399)[\#](#jax.numpy.rint "Link to this definition")  
Rounds the elements of x to the nearest integer

JAX implementation of [`numpy.rint`](https://numpy.org/doc/stable/reference/generated/numpy.rint.html#numpy.rint "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – Input array

Returns:  
An array-like object containing the rounded elements of `x`. Always promotes to inexact.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

If an element of x is exactly half way, e.g. `0.5` or `1.5`, rint will round to the nearest even integer.

Examples

    >>> x1 = jnp.array([5, 4, 7])
    >>> jnp.rint(x1)
    Array([5., 4., 7.], dtype=float32)

    >>> x2 = jnp.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    >>> jnp.rint(x2)
    Array([-2., -2., -0.,  0.,  2.,  2.,  4.,  4.], dtype=float32)

    >>> x3 = jnp.array([-2.5+3.5j, 4.5-0.5j])
    >>> jnp.rint(x3)
    Array([-2.+4.j,  4.-0.j], dtype=complex64)

[](jax.numpy.right_shift.html "previous page")

previous

jax.numpy.right_shift

[](jax.numpy.roll.html "next page")

next

jax.numpy.roll

Contents

- [`rint()`](#jax.numpy.rint)

By The JAX authors

© Copyright 2024, The JAX Authors.\
