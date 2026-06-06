- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.place

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.place.rst "Download source file")
-  .pdf

# jax.numpy.place

## Contents

- [`place()`](#jax.numpy.place)

# jax.numpy.place[\#](#jax-numpy-place "Link to this heading")

jax.numpy.place(*arr*, *mask*, *vals*, *\**, *inplace=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/indexing.py#L1531-L1605)[\#](#jax.numpy.place "Link to this definition")  
Update array elements based on a mask.

JAX implementation of [`numpy.place()`](https://numpy.org/doc/stable/reference/generated/numpy.place.html#numpy.place "(in NumPy v2.4)").

The semantics of [`numpy.place()`](https://numpy.org/doc/stable/reference/generated/numpy.place.html#numpy.place "(in NumPy v2.4)") are to modify arrays in-place, which is not possible for JAX’s immutable arrays. The JAX version returns a modified copy of the input, and adds the `inplace` parameter which must be set to False\` by the user as a reminder of this API difference.

Parameters:  
- **arr** (*ArrayLike*) – array into which values will be placed.

- **mask** (*ArrayLike*) – boolean mask with the same size as `arr`.

- **vals** (*ArrayLike*) – values to be inserted into `arr` at the locations indicated by mask. If too many values are supplied, they will be truncated. If not enough values are supplied, they will be repeated.

- **inplace** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – must be set to False to indicate that the input is not modified in-place, but rather a modified copy is returned.

Returns:  
A copy of `arr` with masked values set to entries from vals.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.put()`](jax.numpy.put.html#jax.numpy.put "jax.numpy.put"): put elements into an array at numerical indices.

- [`jax.numpy.ndarray.at()`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at"): array updates using NumPy-style indexing

Examples

    >>> x = jnp.zeros((3, 5), dtype=int)
    >>> mask = (jnp.arange(x.size) % 3 == 0).reshape(x.shape)
    >>> mask
    Array([[ True, False, False,  True, False],
           [False,  True, False, False,  True],
           [False, False,  True, False, False]], dtype=bool)

Placing a scalar value:

    >>> jnp.place(x, mask, 1, inplace=False)
    Array([[1, 0, 0, 1, 0],
           [0, 1, 0, 0, 1],
           [0, 0, 1, 0, 0]], dtype=int32)

In this case, `jnp.place` is similar to the masked array update syntax:

    >>> x.at[mask].set(1)
    Array([[1, 0, 0, 1, 0],
           [0, 1, 0, 0, 1],
           [0, 0, 1, 0, 0]], dtype=int32)

`place` differs when placing values from an array. The array is repeated to fill the masked entries:

    >>> vals = jnp.array([1, 3, 5])
    >>> jnp.place(x, mask, vals, inplace=False)
    Array([[1, 0, 0, 3, 0],
           [0, 5, 0, 0, 1],
           [0, 0, 3, 0, 0]], dtype=int32)

[](jax.numpy.piecewise.html "previous page")

previous

jax.numpy.piecewise

[](jax.numpy.poly.html "next page")

next

jax.numpy.poly

Contents

- [`place()`](#jax.numpy.place)

By The JAX authors

© Copyright 2024, The JAX Authors.\
