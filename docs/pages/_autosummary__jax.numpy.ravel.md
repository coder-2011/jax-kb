- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.ravel

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.ravel.rst "Download source file")
-  .pdf

# jax.numpy.ravel

## Contents

- [`ravel()`](#jax.numpy.ravel)

# jax.numpy.ravel[\#](#jax-numpy-ravel "Link to this heading")

jax.numpy.ravel(*a*, *order='C'*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L1986-L2038)[\#](#jax.numpy.ravel "Link to this definition")  
Flatten array into a 1-dimensional shape.

JAX implementation of [`numpy.ravel()`](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html#numpy.ravel "(in NumPy v2.4)"), implemented in terms of [`jax.lax.reshape()`](jax.lax.reshape.html#jax.lax.reshape "jax.lax.reshape").

`ravel(arr,`` ``order=order)` is equivalent to `reshape(arr,`` ``-1,`` ``order=order)`.

Parameters:  
- **a** (*ArrayLike*) – array to be flattened.

- **order** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – `'F'` or `'C'`, specifies whether the reshape should apply column-major (fortran-style, `"F"`) or row-major (C-style, `"C"`) order; default is `"C"`. JAX does not support order=”A” or order=”K”.

Returns:  
flattened copy of input array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

Unlike [`numpy.ravel()`](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html#numpy.ravel "(in NumPy v2.4)"), [`jax.numpy.ravel()`](#jax.numpy.ravel "jax.numpy.ravel") will return a copy rather than a view of the input array. However, under JIT, the compiler will optimize-away such copies when possible, so this doesn’t have performance impacts in practice.

See also

- [`jax.Array.ravel()`](jax.Array.ravel.html#jax.Array.ravel "jax.Array.ravel"): equivalent functionality via an array method.

- [`jax.numpy.reshape()`](jax.numpy.reshape.html#jax.numpy.reshape "jax.numpy.reshape"): general array reshape.

Examples

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])

By default, ravel in C-style, row-major order

    >>> jnp.ravel(x)
    Array([1, 2, 3, 4, 5, 6], dtype=int32)

Optionally ravel in Fortran-style, column-major:

    >>> jnp.ravel(x, order='F')
    Array([1, 4, 2, 5, 3, 6], dtype=int32)

For convenience, the same functionality is available via the [`jax.Array.ravel()`](jax.Array.ravel.html#jax.Array.ravel "jax.Array.ravel") method:

    >>> x.ravel()
    Array([1, 2, 3, 4, 5, 6], dtype=int32)

[](jax.numpy.radians.html "previous page")

previous

jax.numpy.radians

[](jax.numpy.ravel_multi_index.html "next page")

next

jax.numpy.ravel_multi_index

Contents

- [`ravel()`](#jax.numpy.ravel)

By The JAX authors

© Copyright 2024, The JAX Authors.\
