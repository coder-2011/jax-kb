- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fill_diagonal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fill_diagonal.rst "Download source file")
-  .pdf

# jax.numpy.fill_diagonal

## Contents

- [`fill_diagonal()`](#jax.numpy.fill_diagonal)

# jax.numpy.fill_diagonal[\#](#jax-numpy-fill-diagonal "Link to this heading")

jax.numpy.fill_diagonal(*a*, *val*, *wrap=False*, *\**, *inplace=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L7073-L7152)[\#](#jax.numpy.fill_diagonal "Link to this definition")  
Return a copy of the array with the diagonal overwritten.

JAX implementation of [`numpy.fill_diagonal()`](https://numpy.org/doc/stable/reference/generated/numpy.fill_diagonal.html#numpy.fill_diagonal "(in NumPy v2.4)").

The semantics of [`numpy.fill_diagonal()`](https://numpy.org/doc/stable/reference/generated/numpy.fill_diagonal.html#numpy.fill_diagonal "(in NumPy v2.4)") are to modify arrays in-place, which is not possible for JAX’s immutable arrays. The JAX version returns a modified copy of the input, and adds the `inplace` parameter which must be set to False\` by the user as a reminder of this API difference.

Parameters:  
- **a** (*ArrayLike*) – input array. Must have `a.ndim`` ``>=`` ``2`. If `a.ndim`` ``>=`` ``3`, then all dimensions must be the same size.

- **val** (*ArrayLike*) – scalar or array with which to fill the diagonal. If an array, it will be flattened and repeated to fill the diagonal entries.

- **wrap** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Not implemented by JAX. Only the default value of `False` is supported.

- **inplace** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – must be set to False to indicate that the input is not modified in-place, but rather a modified copy is returned.

Returns:  
A copy of `a` with the diagonal set to `val`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> x = jnp.zeros((3, 3), dtype=int)
    >>> jnp.fill_diagonal(x, jnp.array([1, 2, 3]), inplace=False)
    Array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3]], dtype=int32)

Unlike [`numpy.fill_diagonal()`](https://numpy.org/doc/stable/reference/generated/numpy.fill_diagonal.html#numpy.fill_diagonal "(in NumPy v2.4)"), the input `x` is not modified.

If the diagonal value has too many entries, it will be truncated

    >>> jnp.fill_diagonal(x, jnp.arange(100, 200), inplace=False)
    Array([[100,   0,   0],
           [  0, 101,   0],
           [  0,   0, 102]], dtype=int32)

If the diagonal has too few entries, it will be repeated:

    >>> x = jnp.zeros((4, 4), dtype=int)
    >>> jnp.fill_diagonal(x, jnp.array([3, 4]), inplace=False)
    Array([[3, 0, 0, 0],
           [0, 4, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]], dtype=int32)

For non-square arrays, the diagonal of the leading square slice is filled:

    >>> x = jnp.zeros((3, 5), dtype=int)
    >>> jnp.fill_diagonal(x, 1, inplace=False)
    Array([[1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0]], dtype=int32)

And for square N-dimensional arrays, the N-dimensional diagonal is filled:

    >>> y = jnp.zeros((2, 2, 2))
    >>> jnp.fill_diagonal(y, 1, inplace=False)
    Array([[[1., 0.],
            [0., 0.]],

           [[0., 0.],
            [0., 1.]]], dtype=float32)

[](jax.numpy.fabs.html "previous page")

previous

jax.numpy.fabs

[](jax.numpy.finfo.html "next page")

next

jax.numpy.finfo

Contents

- [`fill_diagonal()`](#jax.numpy.fill_diagonal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
