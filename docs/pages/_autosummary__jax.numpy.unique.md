- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.unique

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.unique.rst "Download source file")
-  .pdf

# jax.numpy.unique

## Contents

- [`unique()`](#jax.numpy.unique)

# jax.numpy.unique[\#](#jax-numpy-unique "Link to this heading")

jax.numpy.unique(*ar*, *return_index=False*, *return_inverse=False*, *return_counts=False*, *axis=None*, *\**, *equal_nan=True*, *size=None*, *fill_value=None*, *sorted=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/setops.py#L667-L862)[\#](#jax.numpy.unique "Link to this definition")  
Return the unique values from an array.

JAX implementation of [`numpy.unique()`](https://numpy.org/doc/stable/reference/generated/numpy.unique.html#numpy.unique "(in NumPy v2.4)").

Because the size of the output of `unique` is data-dependent, the function is not typically compatible with [`jit()`](jax.jit.html#jax.jit "jax.jit") and other JAX transformations. The JAX version adds the optional `size` argument which must be specified statically for `jnp.unique` to be used in such contexts.

Parameters:  
- **ar** (*ArrayLike*) – N-dimensional array from which unique values will be extracted.

- **return_index** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, also return the indices in `ar` where each value occurs

- **return_inverse** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, also return the indices that can be used to reconstruct `ar` from the unique values.

- **return_counts** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, also return the number of occurrences of each unique value.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – if specified, compute unique values along the specified axis. If None (default), then flatten `ar` before computing the unique values.

- **equal_nan** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, consider NaN values equivalent when determining uniqueness.

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – if specified, return only the first `size` sorted unique elements. If there are fewer unique elements than `size` indicates, the return value will be padded with `fill_value`.

- **fill_value** (*ArrayLike* *\|* *None*) – when `size` is specified and there are fewer than the indicated number of elements, fill the remaining entries `fill_value`. Defaults to the minimum unique value.

- **sorted** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX.

Returns:  
An array or tuple of arrays, depending on the values of `return_index`, `return_inverse`, and `return_counts`. Returned values are

- `unique_values`:  
  if `axis` is None, a 1D array of length `n_unique`, If `axis` is specified, shape is `(*ar.shape[:axis],`` ``n_unique,`` ``*ar.shape[axis`` ``+`` ``1:])`.

- `unique_index`:  
  *(returned only if return_index is True)* An array of shape `(n_unique,)`. Contains the indices of the first occurrence of each unique value in `ar`. For 1D inputs, `ar[unique_index]` is equivalent to `unique_values`.

- `unique_inverse`:  
  *(returned only if return_inverse is True)* An array of shape `(ar.size,)` if `axis` is None, or of shape `(ar.shape[axis],)` if `axis` is specified. Contains the indices within `unique_values` of each value in `ar`. For 1D inputs, `unique_values[unique_inverse]` is equivalent to `ar`.

- `unique_counts`:  
  *(returned only if return_counts is True)* An array of shape `(n_unique,)`. Contains the number of occurrences of each unique value in `ar`.

See also

- [`jax.numpy.unique_counts()`](jax.numpy.unique_counts.html#jax.numpy.unique_counts "jax.numpy.unique_counts"): shortcut to `unique(arr,`` ``return_counts=True)`.

- [`jax.numpy.unique_inverse()`](jax.numpy.unique_inverse.html#jax.numpy.unique_inverse "jax.numpy.unique_inverse"): shortcut to `unique(arr,`` ``return_inverse=True)`.

- [`jax.numpy.unique_all()`](jax.numpy.unique_all.html#jax.numpy.unique_all "jax.numpy.unique_all"): shortcut to `unique` with all return values.

- [`jax.numpy.unique_values()`](jax.numpy.unique_values.html#jax.numpy.unique_values "jax.numpy.unique_values"): like `unique`, but no optional return values.

Examples

    >>> x = jnp.array([3, 4, 1, 3, 1])
    >>> jnp.unique(x)
    Array([1, 3, 4], dtype=int32)

**JIT compilation & the size argument**

If you try this under [`jit()`](jax.jit.html#jax.jit "jax.jit") or another transformation, you will get an error because the output shape is dynamic:

    >>> jax.jit(jnp.unique)(x)  
    Traceback (most recent call last):
       ...
    jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[5].
    The error arose for the first argument of jnp.unique(). To make jnp.unique() compatible with JIT and other transforms, you can specify a concrete value for the size argument, which will determine the output size.

The issue is that the output of transformed functions must have static shapes. In order to make this work, you can pass a static `size` parameter:

    >>> jit_unique = jax.jit(jnp.unique, static_argnames=['size'])
    >>> jit_unique(x, size=3)
    Array([1, 3, 4], dtype=int32)

If your static size is smaller than the true number of unique values, they will be truncated.

    >>> jit_unique(x, size=2)
    Array([1, 3], dtype=int32)

If the static size is larger than the true number of unique values, they will be padded with `fill_value`, which defaults to the minimum unique value:

    >>> jit_unique(x, size=5)
    Array([1, 3, 4, 1, 1], dtype=int32)
    >>> jit_unique(x, size=5, fill_value=0)
    Array([1, 3, 4, 0, 0], dtype=int32)

**Multi-dimensional unique values**

If you pass a multi-dimensional array to `unique`, it will be flattened by default:

    >>> M = jnp.array([[1, 2],
    ...                [2, 3],
    ...                [1, 2]])
    >>> jnp.unique(M)
    Array([1, 2, 3], dtype=int32)

If you pass an `axis` keyword, you can find unique *slices* of the array along that axis:

    >>> jnp.unique(M, axis=0)
    Array([[1, 2],
           [2, 3]], dtype=int32)

**Returning indices**

If you set `return_index=True`, then `unique` returns the indices of the first occurrence of each unique value:

    >>> x = jnp.array([3, 4, 1, 3, 1])
    >>> values, indices = jnp.unique(x, return_index=True)
    >>> print(values)
    [1 3 4]
    >>> print(indices)
    [2 0 1]
    >>> jnp.all(values == x[indices])
    Array(True, dtype=bool)

In multiple dimensions, the unique values can be extracted with [`jax.numpy.take()`](jax.numpy.take.html#jax.numpy.take "jax.numpy.take") evaluated along the specified axis:

    >>> values, indices = jnp.unique(M, axis=0, return_index=True)
    >>> jnp.all(values == jnp.take(M, indices, axis=0))
    Array(True, dtype=bool)

**Returning inverse**

If you set `return_inverse=True`, then `unique` returns the indices within the unique values for every entry in the input array:

    >>> x = jnp.array([3, 4, 1, 3, 1])
    >>> values, inverse = jnp.unique(x, return_inverse=True)
    >>> print(values)
    [1 3 4]
    >>> print(inverse)
    [1 2 0 1 0]
    >>> jnp.all(values[inverse] == x)
    Array(True, dtype=bool)

In multiple dimensions, the input can be reconstructed using [`jax.numpy.take()`](jax.numpy.take.html#jax.numpy.take "jax.numpy.take"):

    >>> values, inverse = jnp.unique(M, axis=0, return_inverse=True)
    >>> jnp.all(jnp.take(values, inverse, axis=0) == M)
    Array(True, dtype=bool)

**Returning counts**

If you set `return_counts=True`, then `unique` returns the number of occurrences within the input for every unique value:

    >>> x = jnp.array([3, 4, 1, 3, 1])
    >>> values, counts = jnp.unique(x, return_counts=True)
    >>> print(values)
    [1 3 4]
    >>> print(counts)
    [2 2 1]

For multi-dimensional arrays, this also returns a 1D array of counts indicating number of occurrences along the specified axis:

    >>> values, counts = jnp.unique(M, axis=0, return_counts=True)
    >>> print(values)
    [[1 2]
     [2 3]]
    >>> print(counts)
    [2 1]

[](jax.numpy.union1d.html "previous page")

previous

jax.numpy.union1d

[](jax.numpy.unique_all.html "next page")

next

jax.numpy.unique_all

Contents

- [`unique()`](#jax.numpy.unique)

By The JAX authors

© Copyright 2024, The JAX Authors.\
