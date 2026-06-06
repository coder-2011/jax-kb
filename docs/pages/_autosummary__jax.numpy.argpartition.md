- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.argpartition

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.argpartition.rst "Download source file")
-  .pdf

# jax.numpy.argpartition

## Contents

- [`argpartition()`](#jax.numpy.argpartition)

# jax.numpy.argpartition[\#](#jax-numpy-argpartition "Link to this heading")

jax.numpy.argpartition(*a*, *kth*, *axis=-1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/sorting.py#L250-L325)[\#](#jax.numpy.argpartition "Link to this definition")  
Returns indices that partially sort an array.

JAX implementation of [`numpy.argpartition()`](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html#numpy.argpartition "(in NumPy v2.4)"). The JAX version differs from NumPy in the treatment of NaN entries: NaNs which have the negative bit set are sorted to the beginning of the array.

Parameters:  
- **a** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – array to be partitioned.

- **kth** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – static integer index about which to partition the array.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – static integer axis along which to partition the array; default is -1.

Returns:  
Indices which partition `a` at the `kth` value along `axis`. The entries before `kth` are indices of values smaller than `take(a,`` ``kth,`` ``axis)`, and entries after `kth` are indices of values larger than `take(a,`` ``kth,`` ``axis)`

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

Note

The JAX version requires the `kth` argument to be a static integer rather than a general array. This is implemented via two calls to [`jax.lax.top_k()`](jax.lax.top_k.html#jax.lax.top_k "jax.lax.top_k"). If you’re only accessing the top or bottom k values of the output, it may be more efficient to call [`jax.lax.top_k()`](jax.lax.top_k.html#jax.lax.top_k "jax.lax.top_k") directly.

See also

- [`jax.numpy.partition()`](jax.numpy.partition.html#jax.numpy.partition "jax.numpy.partition"): direct partial sort

- [`jax.numpy.argsort()`](jax.numpy.argsort.html#jax.numpy.argsort "jax.numpy.argsort"): full indirect sort

- [`jax.lax.top_k()`](jax.lax.top_k.html#jax.lax.top_k "jax.lax.top_k"): directly find the top k entries

- [`jax.lax.approx_max_k()`](jax.lax.approx_max_k.html#jax.lax.approx_max_k "jax.lax.approx_max_k"): compute the approximate top k entries

- [`jax.lax.approx_min_k()`](jax.lax.approx_min_k.html#jax.lax.approx_min_k "jax.lax.approx_min_k"): compute the approximate bottom k entries

Examples

    >>> x = jnp.array([6, 8, 4, 3, 1, 9, 7, 5, 2, 3])
    >>> kth = 4
    >>> idx = jnp.argpartition(x, kth)
    >>> idx
    Array([4, 8, 3, 9, 2, 0, 1, 5, 6, 7], dtype=int32)

The result is a sequence of indices that partially sort the input. All indices before `kth` are of values smaller than the pivot value, and all indices after `kth` are of values larger than the pivot value:

    >>> x_partitioned = x[idx]
    >>> smallest_values = x_partitioned[:kth]
    >>> pivot_value = x_partitioned[kth]
    >>> largest_values = x_partitioned[kth + 1:]
    >>> print(smallest_values, pivot_value, largest_values)
    [1 2 3 3] 4 [6 8 9 7 5]

Notice that among `smallest_values` and `largest_values`, the returned order is arbitrary and implementation-dependent.

[](jax.numpy.argmin.html "previous page")

previous

jax.numpy.argmin

[](jax.numpy.argsort.html "next page")

next

jax.numpy.argsort

Contents

- [`argpartition()`](#jax.numpy.argpartition)

By The JAX authors

© Copyright 2024, The JAX Authors.\
