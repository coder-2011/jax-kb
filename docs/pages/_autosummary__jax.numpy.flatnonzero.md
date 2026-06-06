- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.flatnonzero

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.flatnonzero.rst "Download source file")
-  .pdf

# jax.numpy.flatnonzero

## Contents

- [`flatnonzero()`](#jax.numpy.flatnonzero)

# jax.numpy.flatnonzero[\#](#jax-numpy-flatnonzero "Link to this heading")

jax.numpy.flatnonzero(*a*, *\**, *size=None*, *fill_value=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L3719-L3763)[\#](#jax.numpy.flatnonzero "Link to this definition")  
Return indices of nonzero elements in a flattened array

JAX implementation of [`numpy.flatnonzero()`](https://numpy.org/doc/stable/reference/generated/numpy.flatnonzero.html#numpy.flatnonzero "(in NumPy v2.4)").

`jnp.flatnonzero(x)` is equivalent to `nonzero(ravel(a))[0]`. For a full discussion of the parameters to this function, refer to [`jax.numpy.nonzero()`](jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero").

Parameters:  
- **a** (*ArrayLike*) – N-dimensional array.

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional static integer specifying the number of nonzero entries to return. See [`jax.numpy.nonzero()`](jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero") for more discussion of this parameter.

- **fill_value** (*None* *\|* *ArrayLike* *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[ArrayLike,* *...\]*) – optional padding value when `size` is specified. Defaults to 0. See [`jax.numpy.nonzero()`](jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero") for more discussion of this parameter.

Returns:  
Array containing the indices of each nonzero value in the flattened array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.nonzero()`](jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero")

- [`jax.numpy.where()`](jax.numpy.where.html#jax.numpy.where "jax.numpy.where")

Examples

    >>> x = jnp.array([[0, 5, 0],
    ...                [6, 0, 8]])
    >>> jnp.flatnonzero(x)
    Array([1, 3, 5], dtype=int32)

This is equivalent to calling [`nonzero()`](jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero") on the flattened array, and extracting the first entry in the resulting tuple:

    >>> jnp.nonzero(x.ravel())[0]
    Array([1, 3, 5], dtype=int32)

The returned indices can be used to extract nonzero entries from the flattened array:

    >>> indices = jnp.flatnonzero(x)
    >>> x.ravel()[indices]
    Array([5, 6, 8], dtype=int32)

[](jax.numpy.finfo.html "previous page")

previous

jax.numpy.finfo

[](jax.numpy.flexible.html "next page")

next

jax.numpy.flexible

Contents

- [`flatnonzero()`](#jax.numpy.flatnonzero)

By The JAX authors

© Copyright 2024, The JAX Authors.\
