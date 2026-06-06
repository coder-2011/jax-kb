- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.tril_indices

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.tril_indices.rst "Download source file")
-  .pdf

# jax.numpy.tril_indices

## Contents

- [`tril_indices()`](#jax.numpy.tril_indices)

# jax.numpy.tril_indices[\#](#jax-numpy-tril-indices "Link to this heading")

jax.numpy.tril_indices(*n*, *k=0*, *m=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L6890-L6947)[\#](#jax.numpy.tril_indices "Link to this definition")  
Return the indices of lower triangle of an array of size `(n,`` ``m)`.

JAX implementation of [`numpy.tril_indices()`](https://numpy.org/doc/stable/reference/generated/numpy.tril_indices.html#numpy.tril_indices "(in NumPy v2.4)").

Parameters:  
- **n** (*DimSize*) – int. Number of rows of the array for which the indices are returned.

- **k** (*DimSize*) – optional, int, default=0. Specifies the sub-diagonal on and below which the indices of lower triangle are returned. `k=0` refers to main diagonal, `k<0` refers to sub-diagonal below the main diagonal and `k>0` refers to sub-diagonal above the main diagonal.

- **m** (*DimSize* *\|* *None*) – optional, int. Number of columns of the array for which the indices are returned. If not specified, then `m`` ``=`` ``n`.

Returns:  
A tuple of two arrays containing the indices of the lower triangle, one along each axis.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

See also

- [`jax.numpy.triu_indices()`](jax.numpy.triu_indices.html#jax.numpy.triu_indices "jax.numpy.triu_indices"): Returns the indices of upper triangle of an array of size `(n,`` ``m)`.

- [`jax.numpy.triu_indices_from()`](jax.numpy.triu_indices_from.html#jax.numpy.triu_indices_from "jax.numpy.triu_indices_from"): Returns the indices of upper triangle of a given array.

- [`jax.numpy.tril_indices_from()`](jax.numpy.tril_indices_from.html#jax.numpy.tril_indices_from "jax.numpy.tril_indices_from"): Returns the indices of lower triangle of a given array.

Examples

If only `n` is provided in input, the indices of lower triangle of an array of size `(n,`` ``n)` array are returned.

    >>> jnp.tril_indices(3)
    (Array([0, 1, 1, 2, 2, 2], dtype=int32), Array([0, 0, 1, 0, 1, 2], dtype=int32))

If both `n` and `m` are provided in input, the indices of lower triangle of an `(n,`` ``m)` array are returned.

    >>> jnp.tril_indices(3, m=2)
    (Array([0, 1, 1, 2, 2], dtype=int32), Array([0, 0, 1, 0, 1], dtype=int32))

If `k`` ``=`` ``1`, the indices on and below the first sub-diagonal above the main diagonal are returned.

    >>> jnp.tril_indices(3, k=1)
    (Array([0, 0, 1, 1, 1, 2, 2, 2], dtype=int32), Array([0, 1, 0, 1, 2, 0, 1, 2], dtype=int32))

If `k`` ``=`` ``-1`, the indices on and below the first sub-diagonal below the main diagonal are returned.

    >>> jnp.tril_indices(3, k=-1)
    (Array([1, 2, 2], dtype=int32), Array([0, 0, 1], dtype=int32))

[](jax.numpy.tril.html "previous page")

previous

jax.numpy.tril

[](jax.numpy.tril_indices_from.html "next page")

next

jax.numpy.tril_indices_from

Contents

- [`tril_indices()`](#jax.numpy.tril_indices)

By The JAX authors

© Copyright 2024, The JAX Authors.\
