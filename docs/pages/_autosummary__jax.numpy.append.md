- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.append

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.append.rst "Download source file")
-  .pdf

# jax.numpy.append

## Contents

- [`append()`](#jax.numpy.append)

# jax.numpy.append[\#](#jax-numpy-append "Link to this heading")

jax.numpy.append(*arr*, *values*, *axis=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L7522-L7575)[\#](#jax.numpy.append "Link to this definition")  
Return a new array with values appended to the end of the original array.

JAX implementation of [`numpy.append()`](https://numpy.org/doc/stable/reference/generated/numpy.append.html#numpy.append "(in NumPy v2.4)").

Parameters:  
- **arr** (*ArrayLike*) – original array.

- **values** (*ArrayLike*) – values to be appended to the array. The `values` must have the same number of dimensions as `arr`, and all dimensions must match except in the specified axis.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – axis along which to append values. If None (default), both `arr` and `values` will be flattened before appending.

Returns:  
A new array with values appended to `arr`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.insert()`](jax.numpy.insert.html#jax.numpy.insert "jax.numpy.insert")

- [`jax.numpy.delete()`](jax.numpy.delete.html#jax.numpy.delete "jax.numpy.delete")

Examples

    >>> a = jnp.array([1, 2, 3])
    >>> b = jnp.array([4, 5, 6])
    >>> jnp.append(a, b)
    Array([1, 2, 3, 4, 5, 6], dtype=int32)

Appending along a specific axis:

    >>> a = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> b = jnp.array([[5, 6]])
    >>> jnp.append(a, b, axis=0)
    Array([[1, 2],
           [3, 4],
           [5, 6]], dtype=int32)

Appending along a trailing axis:

    >>> a = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> b = jnp.array([[7], [8]])
    >>> jnp.append(a, b, axis=1)
    Array([[1, 2, 3, 7],
           [4, 5, 6, 8]], dtype=int32)

[](jax.numpy.any.html "previous page")

previous

jax.numpy.any

[](jax.numpy.apply_along_axis.html "next page")

next

jax.numpy.apply_along_axis

Contents

- [`append()`](#jax.numpy.append)

By The JAX authors

© Copyright 2024, The JAX Authors.\
