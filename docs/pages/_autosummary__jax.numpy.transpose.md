- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.transpose

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.transpose.rst "Download source file")
-  .pdf

# jax.numpy.transpose

## Contents

- [`transpose()`](#jax.numpy.transpose)

# jax.numpy.transpose[\#](#jax-numpy-transpose "Link to this heading")

jax.numpy.transpose(*a*, *axes=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L1088-L1165)[\#](#jax.numpy.transpose "Link to this definition")  
Return a transposed version of an N-dimensional array.

JAX implementation of [`numpy.transpose()`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html#numpy.transpose "(in NumPy v2.4)"), implemented in terms of [`jax.lax.transpose()`](jax.lax.transpose.html#jax.lax.transpose "jax.lax.transpose").

Parameters:  
- **a** (*ArrayLike*) – input array

- **axes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – optionally specify the permutation using a length-a.ndim sequence of integers `i` satisfying `0`` ``<=`` ``i`` ``<`` ``a.ndim`. Defaults to `range(a.ndim)[::-1]`, i.e. reverses the order of all axes.

Returns:  
transposed copy of the array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.Array.transpose()`](jax.Array.transpose.html#jax.Array.transpose "jax.Array.transpose"): equivalent function via an [`Array`](jax.Array.html#jax.Array "jax.Array") method.

- [`jax.Array.T`](jax.Array.T.html#jax.Array.T "jax.Array.T"): equivalent function via an [`Array`](jax.Array.html#jax.Array "jax.Array") property.

- [`jax.numpy.matrix_transpose()`](jax.numpy.matrix_transpose.html#jax.numpy.matrix_transpose "jax.numpy.matrix_transpose"): transpose the last two axes of an array. This is suitable for working with batched 2D matrices.

- [`jax.numpy.swapaxes()`](jax.numpy.swapaxes.html#jax.numpy.swapaxes "jax.numpy.swapaxes"): swap any two axes in an array.

- [`jax.numpy.moveaxis()`](jax.numpy.moveaxis.html#jax.numpy.moveaxis "jax.numpy.moveaxis"): move an axis to another position in the array.

Note

Unlike [`numpy.transpose()`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html#numpy.transpose "(in NumPy v2.4)"), [`jax.numpy.transpose()`](#jax.numpy.transpose "jax.numpy.transpose") will return a copy rather than a view of the input array. However, under JIT, the compiler will optimize-away such copies when possible, so this doesn’t have performance impacts in practice.

Examples

For a 1D array, the transpose is the identity:

    >>> x = jnp.array([1, 2, 3, 4])
    >>> jnp.transpose(x)
    Array([1, 2, 3, 4], dtype=int32)

For a 2D array, the transpose is a matrix transpose:

    >>> x = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> jnp.transpose(x)
    Array([[1, 3],
           [2, 4]], dtype=int32)

For an N-dimensional array, the transpose reverses the order of the axes:

    >>> x = jnp.zeros(shape=(3, 4, 5))
    >>> jnp.transpose(x).shape
    (5, 4, 3)

The `axes` argument can be specified to change this default behavior:

    >>> jnp.transpose(x, (0, 2, 1)).shape
    (3, 5, 4)

Since swapping the last two axes is a common operation, it can be done via its own API, [`jax.numpy.matrix_transpose()`](jax.numpy.matrix_transpose.html#jax.numpy.matrix_transpose "jax.numpy.matrix_transpose"):

    >>> jnp.matrix_transpose(x).shape
    (3, 5, 4)

For convenience, transposes may also be performed using the [`jax.Array.transpose()`](jax.Array.transpose.html#jax.Array.transpose "jax.Array.transpose") method or the [`jax.Array.T`](jax.Array.T.html#jax.Array.T "jax.Array.T") property:

    >>> x = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> x.transpose()
    Array([[1, 3],
           [2, 4]], dtype=int32)
    >>> x.T
    Array([[1, 3],
           [2, 4]], dtype=int32)

[](jax.numpy.trapezoid.html "previous page")

previous

jax.numpy.trapezoid

[](jax.numpy.tri.html "next page")

next

jax.numpy.tri

Contents

- [`transpose()`](#jax.numpy.transpose)

By The JAX authors

© Copyright 2024, The JAX Authors.\
