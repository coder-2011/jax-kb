- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.searchsorted

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.searchsorted.rst "Download source file")
-  .pdf

# jax.numpy.searchsorted

## Contents

- [`searchsorted()`](#jax.numpy.searchsorted)

# jax.numpy.searchsorted[\#](#jax-numpy-searchsorted "Link to this heading")

jax.numpy.searchsorted(*a*, *v*, *side='left'*, *sorter=None*, *\**, *method='scan'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L9326-L9401)[\#](#jax.numpy.searchsorted "Link to this definition")  
Perform a binary search within a sorted array.

JAX implementation of [`numpy.searchsorted()`](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html#numpy.searchsorted "(in NumPy v2.4)").

This will return the indices within a sorted array `a` where values in `v` can be inserted to maintain its sort order.

Parameters:  
- **a** (*ArrayLike*) – one-dimensional array, assumed to be in sorted order unless `sorter` is specified.

- **v** (*ArrayLike*) – N-dimensional array of query values

- **side** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – `'left'` (default) or `'right'`; specifies whether insertion indices will be to the left or the right in case of ties.

- **sorter** (*ArrayLike* *\|* *None*) – optional array of indices specifying the sort order of `a`. If specified, then the algorithm assumes that `a[sorter]` is in sorted order.

- **method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – one of `'scan'` (default), `'scan_unrolled'`, `'sort'` or `'compare_all'`. See *Note* below.

Returns:  
Array of insertion indices of shape `v.shape`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

The `method` argument controls the algorithm used to compute the insertion indices.

- `'scan'` (the default) tends to be more performant on CPU, particularly when `a` is very large.

- `'scan_unrolled'` is more performant on GPU at the expense of additional compile time.

- `'sort'` is often more performant on accelerator backends like GPU and TPU, particularly when `v` is very large.

- `'compare_all'` tends to be the most performant when `a` is very small.

Examples

Searching for a single value:

    >>> a = jnp.array([1, 2, 2, 3, 4, 5, 5])
    >>> jnp.searchsorted(a, 2)
    Array(1, dtype=int32)
    >>> jnp.searchsorted(a, 2, side='right')
    Array(3, dtype=int32)

Searching for a batch of values:

    >>> vals = jnp.array([0, 3, 8, 1.5, 2])
    >>> jnp.searchsorted(a, vals)
    Array([0, 3, 7, 1, 1], dtype=int32)

Optionally, the `sorter` argument can be used to find insertion indices into an array sorted via [`jax.numpy.argsort()`](jax.numpy.argsort.html#jax.numpy.argsort "jax.numpy.argsort"):

    >>> a = jnp.array([4, 3, 5, 1, 2])
    >>> sorter = jnp.argsort(a)
    >>> jnp.searchsorted(a, vals, sorter=sorter)
    Array([0, 2, 5, 1, 1], dtype=int32)

The result is equivalent to passing the sorted array:

    >>> jnp.searchsorted(jnp.sort(a), vals)
    Array([0, 2, 5, 1, 1], dtype=int32)

[](jax.numpy.savez.html "previous page")

previous

jax.numpy.savez

[](jax.numpy.select.html "next page")

next

jax.numpy.select

Contents

- [`searchsorted()`](#jax.numpy.searchsorted)

By The JAX authors

© Copyright 2024, The JAX Authors.\
