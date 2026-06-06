- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.argsort

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.argsort.rst "Download source file")
-  .pdf

# jax.numpy.argsort

## Contents

- [`argsort()`](#jax.numpy.argsort)

# jax.numpy.argsort[\#](#jax-numpy-argsort "Link to this heading")

jax.numpy.argsort(*a*, *axis=-1*, *\**, *kind=None*, *order=None*, *stable=True*, *descending=False*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/sorting.py#L91-L178)[\#](#jax.numpy.argsort "Link to this definition")  
Return indices that sort an array.

JAX implementation of [`numpy.argsort()`](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html#numpy.argsort "(in NumPy v2.4)").

Parameters:  
- **a** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – array to sort

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – integer axis along which to sort. Defaults to `-1`, i.e. the last axis. If `None`, then `a` is flattened before being sorted.

- **stable** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – boolean specifying whether a stable sort should be used. Default=True.

- **descending** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – boolean specifying whether to sort in descending order. Default=False.

- **kind** (*None*) – deprecated; instead specify sort algorithm using stable=True or stable=False.

- **order** (*None*) – not supported by JAX

- **dtype** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* [*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") *\|* *SupportsDType* *\|* *None*) – optionally specify the dtype of the resulting indices. If not specified, the default integer dtype will be used.

Returns:  
Array of indices that sort an array. Returned array will be of shape `a.shape` (if `axis` is an integer) or of shape `(a.size,)` (if `axis` is None).

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

Examples

Simple 1-dimensional sort

    >>> x = jnp.array([1, 3, 5, 4, 2, 1])
    >>> indices = jnp.argsort(x)
    >>> indices
    Array([0, 5, 4, 1, 3, 2], dtype=int32)
    >>> x[indices]
    Array([1, 1, 2, 3, 4, 5], dtype=int32)

Sort along the last axis of an array:

    >>> x = jnp.array([[2, 1, 3],
    ...                [6, 4, 3]])
    >>> indices = jnp.argsort(x, axis=1)
    >>> indices
    Array([[1, 0, 2],
           [2, 1, 0]], dtype=int32)
    >>> jnp.take_along_axis(x, indices, axis=1)
    Array([[1, 2, 3],
           [3, 4, 6]], dtype=int32)

See also

- [`jax.numpy.sort()`](jax.numpy.sort.html#jax.numpy.sort "jax.numpy.sort"): return sorted values directly.

- [`jax.numpy.lexsort()`](jax.numpy.lexsort.html#jax.numpy.lexsort "jax.numpy.lexsort"): lexicographical sort of multiple arrays.

- [`jax.lax.sort()`](jax.lax.sort.html#jax.lax.sort "jax.lax.sort"): lower-level function wrapping XLA’s Sort operator.

[](jax.numpy.argpartition.html "previous page")

previous

jax.numpy.argpartition

[](jax.numpy.argwhere.html "next page")

next

jax.numpy.argwhere

Contents

- [`argsort()`](#jax.numpy.argsort)

By The JAX authors

© Copyright 2024, The JAX Authors.\
