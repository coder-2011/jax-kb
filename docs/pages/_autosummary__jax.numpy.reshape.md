- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.reshape

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.reshape.rst "Download source file")
-  .pdf

# jax.numpy.reshape

## Contents

- [`reshape()`](#jax.numpy.reshape)

# jax.numpy.reshape[\#](#jax-numpy-reshape "Link to this heading")

jax.numpy.reshape(*a*, *shape*, *order='C'*, *\**, *copy=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L1901-L1984)[\#](#jax.numpy.reshape "Link to this definition")  
Return a reshaped copy of an array.

JAX implementation of [`numpy.reshape()`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy.reshape "(in NumPy v2.4)"), implemented in terms of [`jax.lax.reshape()`](jax.lax.reshape.html#jax.lax.reshape "jax.lax.reshape").

Parameters:  
- **a** (*ArrayLike*) – input array to reshape

- **shape** (*DimSize* *\|* *Shape*) – integer or sequence of integers giving the new shape, which must match the size of the input array. If any single dimension is given size `-1`, it will be replaced with a value such that the output has the correct size.

- **order** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – `'F'` or `'C'`, specifies whether the reshape should apply column-major (fortran-style, `"F"`) or row-major (C-style, `"C"`) order; default is `"C"`. JAX does not support `order="A"`.

- **copy** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None*) – unused by JAX; JAX always returns a copy, though under JIT the compiler may optimize such copies away.

Returns:  
reshaped copy of input array with the specified shape.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

Unlike [`numpy.reshape()`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy.reshape "(in NumPy v2.4)"), [`jax.numpy.reshape()`](#jax.numpy.reshape "jax.numpy.reshape") will return a copy rather than a view of the input array. However, under JIT, the compiler will optimize-away such copies when possible, so this doesn’t have performance impacts in practice.

See also

- [`jax.Array.reshape()`](jax.Array.reshape.html#jax.Array.reshape "jax.Array.reshape"): equivalent functionality via an array method.

- [`jax.numpy.ravel()`](jax.numpy.ravel.html#jax.numpy.ravel "jax.numpy.ravel"): flatten an array into a 1D shape.

- [`jax.numpy.squeeze()`](jax.numpy.squeeze.html#jax.numpy.squeeze "jax.numpy.squeeze"): remove one or more length-1 axes from an array’s shape.

Examples

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> jnp.reshape(x, 6)
    Array([1, 2, 3, 4, 5, 6], dtype=int32)
    >>> jnp.reshape(x, (3, 2))
    Array([[1, 2],
           [3, 4],
           [5, 6]], dtype=int32)

You can use `-1` to automatically compute a shape that is consistent with the input size:

    >>> jnp.reshape(x, -1)  # -1 is inferred to be 6
    Array([1, 2, 3, 4, 5, 6], dtype=int32)
    >>> jnp.reshape(x, (-1, 2))  # -1 is inferred to be 3
    Array([[1, 2],
           [3, 4],
           [5, 6]], dtype=int32)

The default ordering of axes in the reshape is C-style row-major ordering. To use Fortran-style column-major ordering, specify `order='F'`:

    >>> jnp.reshape(x, 6, order='F')
    Array([1, 4, 2, 5, 3, 6], dtype=int32)
    >>> jnp.reshape(x, (3, 2), order='F')
    Array([[1, 5],
           [4, 3],
           [2, 6]], dtype=int32)

For convenience, this functionality is also available via the [`jax.Array.reshape()`](jax.Array.reshape.html#jax.Array.reshape "jax.Array.reshape") method:

    >>> x.reshape(3, 2)
    Array([[1, 2],
           [3, 4],
           [5, 6]], dtype=int32)

[](jax.numpy.repeat.html "previous page")

previous

jax.numpy.repeat

[](jax.numpy.resize.html "next page")

next

jax.numpy.resize

Contents

- [`reshape()`](#jax.numpy.reshape)

By The JAX authors

© Copyright 2024, The JAX Authors.\
