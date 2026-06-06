- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.lexsort

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.lexsort.rst "Download source file")
-  .pdf

# jax.numpy.lexsort

## Contents

- [`lexsort()`](#jax.numpy.lexsort)

# jax.numpy.lexsort[\#](#jax-numpy-lexsort "Link to this heading")

jax.numpy.lexsort(*keys*, *axis=-1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/sorting.py#L365-L444)[\#](#jax.numpy.lexsort "Link to this definition")  
Sort a sequence of keys in lexicographic order.

JAX implementation of [`numpy.lexsort()`](https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html#numpy.lexsort "(in NumPy v2.4)").

Parameters:  
- **keys** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")*\[*[*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")*\]*) – a sequence of arrays to sort; all arrays must have the same shape. The last key in the sequence is used as the primary key.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the axis along which to sort (default: -1).

Returns:  
An array of integers of shape `keys[0].shape` giving the indices of the entries in lexicographically-sorted order.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.argsort()`](jax.numpy.argsort.html#jax.numpy.argsort "jax.numpy.argsort"): sort a single entry by index.

- [`jax.lax.sort()`](jax.lax.sort.html#jax.lax.sort "jax.lax.sort"): direct XLA sorting API.

Examples

[`lexsort()`](#jax.numpy.lexsort "jax.numpy.lexsort") with a single key is equivalent to [`argsort()`](jax.numpy.argsort.html#jax.numpy.argsort "jax.numpy.argsort"):

    >>> key1 = jnp.array([4, 2, 3, 2, 5])
    >>> jnp.lexsort([key1])
    Array([1, 3, 2, 0, 4], dtype=int32)
    >>> jnp.argsort(key1)
    Array([1, 3, 2, 0, 4], dtype=int32)

With multiple keys, [`lexsort()`](#jax.numpy.lexsort "jax.numpy.lexsort") uses the last key as the primary key:

    >>> key2 = jnp.array([2, 1, 1, 2, 2])
    >>> jnp.lexsort([key1, key2])
    Array([1, 2, 3, 0, 4], dtype=int32)

The meaning of the indices become more clear when printing the sorted keys:

    >>> indices = jnp.lexsort([key1, key2])
    >>> print(f"{key1[indices]}\n{key2[indices]}")
    [2 3 2 4 5]
    [1 1 2 2 2]

Notice that the elements of `key2` appear in order, and within the sequences of duplicated values the corresponding elements of `` `key1 `` appear in order.

For multi-dimensional inputs, [`lexsort()`](#jax.numpy.lexsort "jax.numpy.lexsort") defaults to sorting along the last axis:

    >>> key1 = jnp.array([[2, 4, 2, 3],
    ...                   [3, 1, 2, 2]])
    >>> key2 = jnp.array([[1, 2, 1, 3],
    ...                   [2, 1, 2, 1]])
    >>> jnp.lexsort([key1, key2])
    Array([[0, 2, 1, 3],
           [1, 3, 2, 0]], dtype=int32)

A different sort axis can be chosen using the `axis` keyword; here we sort along the leading axis:

    >>> jnp.lexsort([key1, key2], axis=0)
    Array([[0, 1, 0, 1],
           [1, 0, 1, 0]], dtype=int32)

[](jax.numpy.less_equal.html "previous page")

previous

jax.numpy.less_equal

[](jax.numpy.linspace.html "next page")

next

jax.numpy.linspace

Contents

- [`lexsort()`](#jax.numpy.lexsort)

By The JAX authors

© Copyright 2024, The JAX Authors.\
