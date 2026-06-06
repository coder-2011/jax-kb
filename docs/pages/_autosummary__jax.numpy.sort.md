- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.sort

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.sort.rst "Download source file")
-  .pdf

# jax.numpy.sort

## Contents

- [`sort()`](#jax.numpy.sort)

# jax.numpy.sort[\#](#jax-numpy-sort "Link to this heading")

jax.numpy.sort(*a*, *axis=-1*, *\**, *kind=None*, *order=None*, *stable=True*, *descending=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/sorting.py#L30-L90)[\#](#jax.numpy.sort "Link to this definition")  
Return a sorted copy of an array.

JAX implementation of [`numpy.sort()`](https://numpy.org/doc/stable/reference/generated/numpy.sort.html#numpy.sort "(in NumPy v2.4)").

Parameters:  
- **a** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – array to sort

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – integer axis along which to sort. Defaults to `-1`, i.e. the last axis. If `None`, then `a` is flattened before being sorted.

- **stable** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – boolean specifying whether a stable sort should be used. Default=True.

- **descending** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – boolean specifying whether to sort in descending order. Default=False.

- **kind** (*None*) – deprecated; instead specify sort algorithm using stable=True or stable=False.

- **order** (*None*) – not supported by JAX

Returns:  
Sorted array of shape `a.shape` (if `axis` is an integer) or of shape `(a.size,)` (if `axis` is None).

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

Examples

Simple 1-dimensional sort

    >>> x = jnp.array([1, 3, 5, 4, 2, 1])
    >>> jnp.sort(x)
    Array([1, 1, 2, 3, 4, 5], dtype=int32)

Sort along the last axis of an array:

    >>> x = jnp.array([[2, 1, 3],
    ...                [4, 3, 6]])
    >>> jnp.sort(x, axis=1)
    Array([[1, 2, 3],
           [3, 4, 6]], dtype=int32)

See also

- [`jax.numpy.argsort()`](jax.numpy.argsort.html#jax.numpy.argsort "jax.numpy.argsort"): return indices of sorted values.

- [`jax.numpy.lexsort()`](jax.numpy.lexsort.html#jax.numpy.lexsort "jax.numpy.lexsort"): lexicographical sort of multiple arrays.

- [`jax.lax.sort()`](jax.lax.sort.html#jax.lax.sort "jax.lax.sort"): lower-level function wrapping XLA’s Sort operator.

[](jax.numpy.size.html "previous page")

previous

jax.numpy.size

[](jax.numpy.sort_complex.html "next page")

next

jax.numpy.sort_complex

Contents

- [`sort()`](#jax.numpy.sort)

By The JAX authors

© Copyright 2024, The JAX Authors.\
