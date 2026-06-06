- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.insert

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.insert.rst "Download source file")
-  .pdf

# jax.numpy.insert

## Contents

- [`insert()`](#jax.numpy.insert)

# jax.numpy.insert[\#](#jax-numpy-insert "Link to this heading")

jax.numpy.insert(*arr*, *obj*, *values*, *axis=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L7709-L7805)[\#](#jax.numpy.insert "Link to this definition")  
Insert entries into an array at specified indices.

JAX implementation of [`numpy.insert()`](https://numpy.org/doc/stable/reference/generated/numpy.insert.html#numpy.insert "(in NumPy v2.4)").

Parameters:  
- **arr** (*ArrayLike*) – array object into which values will be inserted.

- **obj** (*ArrayLike* *\|* [*slice*](https://docs.python.org/3/library/functions.html#slice "(in Python v3.14)")) – slice or array of indices specifying insertion locations.

- **values** (*ArrayLike*) – array of values to be inserted.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – specify the insertion axis in the case of multi-dimensional arrays. If unspecified, `arr` will be flattened.

Returns:  
A copy of `arr` with values inserted at the specified locations.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.delete()`](jax.numpy.delete.html#jax.numpy.delete "jax.numpy.delete"): delete entries from an array.

Examples

Inserting a single value:

    >>> x = jnp.arange(5)
    >>> jnp.insert(x, 2, 99)
    Array([ 0,  1, 99,  2,  3,  4], dtype=int32)

Inserting multiple identical values using a slice:

    >>> jnp.insert(x, slice(None, None, 2), -1)
    Array([-1,  0,  1, -1,  2,  3, -1,  4], dtype=int32)

Inserting multiple values using an index:

    >>> indices = jnp.array([4, 2, 5])
    >>> values = jnp.array([10, 11, 12])
    >>> jnp.insert(x, indices, values)
    Array([ 0,  1, 11,  2,  3, 10,  4, 12], dtype=int32)

Inserting columns into a 2D array:

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> indices = jnp.array([1, 3])
    >>> values = jnp.array([[10, 11],
    ...                     [12, 13]])
    >>> jnp.insert(x, indices, values, axis=1)
    Array([[ 1, 10,  2,  3, 11],
           [ 4, 12,  5,  6, 13]], dtype=int32)

[](jax.numpy.inner.html "previous page")

previous

jax.numpy.inner

[](jax.numpy.int_.html "next page")

next

jax.numpy.int\_

Contents

- [`insert()`](#jax.numpy.insert)

By The JAX authors

© Copyright 2024, The JAX Authors.\
