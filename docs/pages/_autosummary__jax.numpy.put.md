- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.put

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.put.rst "Download source file")
-  .pdf

# jax.numpy.put

## Contents

- [`put()`](#jax.numpy.put)

# jax.numpy.put[\#](#jax-numpy-put "Link to this heading")

jax.numpy.put(*a*, *ind*, *v*, *mode=None*, *\**, *inplace=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/indexing.py#L1607-L1695)[\#](#jax.numpy.put "Link to this definition")  
Put elements into an array at given indices.

JAX implementation of [`numpy.put()`](https://numpy.org/doc/stable/reference/generated/numpy.put.html#numpy.put "(in NumPy v2.4)").

The semantics of [`numpy.put()`](https://numpy.org/doc/stable/reference/generated/numpy.put.html#numpy.put "(in NumPy v2.4)") are to modify arrays in-place, which is not possible for JAX’s immutable arrays. The JAX version returns a modified copy of the input, and adds the `inplace` parameter which must be set to False\` by the user as a reminder of this API difference.

Parameters:  
- **a** (*ArrayLike*) – array into which values will be placed.

- **ind** (*ArrayLike*) – array of indices over the flattened array at which to put values.

- **v** (*ArrayLike*) – array of values to put into the array.

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) –

  string specifying how to handle out-of-bound indices. Supported values:

  - `"clip"` (default): clip out-of-bound indices to the final index.

  - `"wrap"`: wrap out-of-bound indices to the beginning of the array.

- **inplace** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – must be set to False to indicate that the input is not modified in-place, but rather a modified copy is returned.

Returns:  
A copy of `a` with specified entries updated.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.place()`](jax.numpy.place.html#jax.numpy.place "jax.numpy.place"): place elements into an array via boolean mask.

- [`jax.numpy.ndarray.at()`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at"): array updates using NumPy-style indexing.

- [`jax.numpy.take()`](jax.numpy.take.html#jax.numpy.take "jax.numpy.take"): extract values from an array at given indices.

Examples

    >>> x = jnp.zeros(5, dtype=int)
    >>> indices = jnp.array([0, 2, 4])
    >>> values = jnp.array([10, 20, 30])
    >>> jnp.put(x, indices, values, inplace=False)
    Array([10,  0, 20,  0, 30], dtype=int32)

This is equivalent to the following [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at") indexing syntax:

    >>> x.at[indices].set(values)
    Array([10,  0, 20,  0, 30], dtype=int32)

There are two modes for handling out-of-bound indices. By default they are clipped:

    >>> indices = jnp.array([0, 2, 6])
    >>> jnp.put(x, indices, values, inplace=False, mode='clip')
    Array([10,  0, 20,  0, 30], dtype=int32)

Alternatively, they can be wrapped to the beginning of the array:

    >>> jnp.put(x, indices, values, inplace=False, mode='wrap')
    Array([10,  30, 20,  0, 0], dtype=int32)

For N-dimensional inputs, the indices refer to the flattened array:

    >>> x = jnp.zeros((3, 5), dtype=int)
    >>> indices = jnp.array([0, 7, 14])
    >>> jnp.put(x, indices, values, inplace=False)
    Array([[10,  0,  0,  0,  0],
           [ 0,  0, 20,  0,  0],
           [ 0,  0,  0,  0, 30]], dtype=int32)

[](jax.numpy.ptp.html "previous page")

previous

jax.numpy.ptp

[](jax.numpy.put_along_axis.html "next page")

next

jax.numpy.put_along_axis

Contents

- [`put()`](#jax.numpy.put)

By The JAX authors

© Copyright 2024, The JAX Authors.\
