- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.partition

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.partition.rst "Download source file")
-  .pdf

# jax.numpy.partition

## Contents

- [`partition()`](#jax.numpy.partition)

# jax.numpy.partition[\#](#jax-numpy-partition "Link to this heading")

jax.numpy.partition(*a*, *kth*, *axis=-1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/sorting.py#L180-L248)[\#](#jax.numpy.partition "Link to this definition")  
Returns a partially-sorted copy of an array.

JAX implementation of [`numpy.partition()`](https://numpy.org/doc/stable/reference/generated/numpy.partition.html#numpy.partition "(in NumPy v2.4)"). The JAX version differs from NumPy in the treatment of NaN entries: NaNs which have the negative bit set are sorted to the beginning of the array.

Parameters:  
- **a** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – array to be partitioned.

- **kth** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – static integer index about which to partition the array.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – static integer axis along which to partition the array; default is -1.

Returns:  
A copy of `a` partitioned at the `kth` value along `axis`. The entries before `kth` are values smaller than `take(a,`` ``kth,`` ``axis)`, and entries after `kth` are indices of values larger than `take(a,`` ``kth,`` ``axis)`

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

Note

The JAX version requires the `kth` argument to be a static integer rather than a general array. This is implemented via two calls to [`jax.lax.top_k()`](jax.lax.top_k.html#jax.lax.top_k "jax.lax.top_k"). If you’re only accessing the top or bottom k values of the output, it may be more efficient to call [`jax.lax.top_k()`](jax.lax.top_k.html#jax.lax.top_k "jax.lax.top_k") directly.

See also

- [`jax.numpy.sort()`](jax.numpy.sort.html#jax.numpy.sort "jax.numpy.sort"): full sort

- [`jax.numpy.argpartition()`](jax.numpy.argpartition.html#jax.numpy.argpartition "jax.numpy.argpartition"): indirect partial sort

- [`jax.lax.top_k()`](jax.lax.top_k.html#jax.lax.top_k "jax.lax.top_k"): directly find the top k entries

- [`jax.lax.approx_max_k()`](jax.lax.approx_max_k.html#jax.lax.approx_max_k "jax.lax.approx_max_k"): compute the approximate top k entries

- [`jax.lax.approx_min_k()`](jax.lax.approx_min_k.html#jax.lax.approx_min_k "jax.lax.approx_min_k"): compute the approximate bottom k entries

Examples

    >>> x = jnp.array([6, 8, 4, 3, 1, 9, 7, 5, 2, 3])
    >>> kth = 4
    >>> x_partitioned = jnp.partition(x, kth)
    >>> x_partitioned
    Array([1, 2, 3, 3, 4, 9, 8, 7, 6, 5], dtype=int32)

The result is a partially-sorted copy of the input. All values before `kth` are of smaller than the pivot value, and all values after `kth` are larger than the pivot value:

    >>> smallest_values = x_partitioned[:kth]
    >>> pivot_value = x_partitioned[kth]
    >>> largest_values = x_partitioned[kth + 1:]
    >>> print(smallest_values, pivot_value, largest_values)
    [1 2 3 3] 4 [9 8 7 6 5]

Notice that among `smallest_values` and `largest_values`, the returned order is arbitrary and implementation-dependent.

[](jax.numpy.pad.html "previous page")

previous

jax.numpy.pad

[](jax.numpy.percentile.html "next page")

next

jax.numpy.percentile

Contents

- [`partition()`](#jax.numpy.partition)

By The JAX authors

© Copyright 2024, The JAX Authors.\
