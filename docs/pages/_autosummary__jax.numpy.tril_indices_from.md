- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.tril_indices_from

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.tril_indices_from.rst "Download source file")
-  .pdf

# jax.numpy.tril_indices_from

## Contents

- [`tril_indices_from()`](#jax.numpy.tril_indices_from)

# jax.numpy.tril_indices_from[\#](#jax-numpy-tril-indices-from "Link to this heading")

jax.numpy.tril_indices_from(*arr*, *k=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L7011-L7071)[\#](#jax.numpy.tril_indices_from "Link to this definition")  
Return the indices of lower triangle of a given array.

JAX implementation of [`numpy.tril_indices_from()`](https://numpy.org/doc/stable/reference/generated/numpy.tril_indices_from.html#numpy.tril_indices_from "(in NumPy v2.4)").

Parameters:  
- **arr** (*ArrayLike* *\|* *SupportsShape*) – input array. Must have `arr.ndim`` ``==`` ``2`.

- **k** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – optional, int, default=0. Specifies the sub-diagonal on and below which the indices of upper triangle are returned. `k=0` refers to main diagonal, `k<0` refers to sub-diagonal below the main diagonal and `k>0` refers to sub-diagonal above the main diagonal.

Returns:  
A tuple of two arrays containing the indices of the lower triangle, one along each axis.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

See also

- [`jax.numpy.triu_indices_from()`](jax.numpy.triu_indices_from.html#jax.numpy.triu_indices_from "jax.numpy.triu_indices_from"): Returns the indices of upper triangle of a given array.

- [`jax.numpy.tril_indices()`](jax.numpy.tril_indices.html#jax.numpy.tril_indices "jax.numpy.tril_indices"): Returns the indices of lower triangle of an array of size `(n,`` ``m)`.

- [`jax.numpy.tril()`](jax.numpy.tril.html#jax.numpy.tril "jax.numpy.tril"): Returns a lower triangle of an array

Examples

    >>> arr = jnp.array([[1, 2, 3],
    ...                  [4, 5, 6],
    ...                  [7, 8, 9]])
    >>> jnp.tril_indices_from(arr)
    (Array([0, 1, 1, 2, 2, 2], dtype=int32), Array([0, 0, 1, 0, 1, 2], dtype=int32))

Elements indexed by `jnp.tril_indices_from` correspond to those in the output of `jnp.tril`.

    >>> ind = jnp.tril_indices_from(arr)
    >>> arr[ind]
    Array([1, 4, 5, 7, 8, 9], dtype=int32)
    >>> jnp.tril(arr)
    Array([[1, 0, 0],
           [4, 5, 0],
           [7, 8, 9]], dtype=int32)

When `k`` ``>`` ``0`:

    >>> jnp.tril_indices_from(arr, k=1)
    (Array([0, 0, 1, 1, 1, 2, 2, 2], dtype=int32), Array([0, 1, 0, 1, 2, 0, 1, 2], dtype=int32))

When `k`` ``<`` ``0`:

    >>> jnp.tril_indices_from(arr, k=-1)
    (Array([1, 2, 2], dtype=int32), Array([0, 0, 1], dtype=int32))

[](jax.numpy.tril_indices.html "previous page")

previous

jax.numpy.tril_indices

[](jax.numpy.trim_zeros.html "next page")

next

jax.numpy.trim_zeros

Contents

- [`tril_indices_from()`](#jax.numpy.tril_indices_from)

By The JAX authors

© Copyright 2024, The JAX Authors.\
