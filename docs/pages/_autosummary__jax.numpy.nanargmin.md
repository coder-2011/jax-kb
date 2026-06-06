- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.nanargmin

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.nanargmin.rst "Download source file")
-  .pdf

# jax.numpy.nanargmin

## Contents

- [`nanargmin()`](#jax.numpy.nanargmin)

# jax.numpy.nanargmin[\#](#jax-numpy-nanargmin "Link to this heading")

jax.numpy.nanargmin(*a*, *axis=None*, *out=None*, *keepdims=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L8408-L8457)[\#](#jax.numpy.nanargmin "Link to this definition")  
Return the index of the minimum value of an array, ignoring NaNs.

JAX implementation of [`numpy.nanargmin()`](https://numpy.org/doc/stable/reference/generated/numpy.nanargmin.html#numpy.nanargmin "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional integer specifying the axis along which to find the maximum value. If `axis` is not specified, `a` will be flattened.

- **out** (*None*) – unused by JAX

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None*) – if True, then return an array with the same number of dimensions as `a`.

Returns:  
an array containing the index of the minimum value along the specified axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

In the case of an axis with all-NaN values, the returned index will be -1. This differs from the behavior of [`numpy.nanargmin()`](https://numpy.org/doc/stable/reference/generated/numpy.nanargmin.html#numpy.nanargmin "(in NumPy v2.4)"), which raises an error.

See also

- [`jax.numpy.argmin()`](jax.numpy.argmin.html#jax.numpy.argmin "jax.numpy.argmin"): return the index of the minimum value.

- [`jax.numpy.nanargmax()`](jax.numpy.nanargmax.html#jax.numpy.nanargmax "jax.numpy.nanargmax"): compute `argmax` while ignoring NaN values.

Examples

    >>> x = jnp.array([jnp.nan, 3, 5, 4, 2])
    >>> jnp.nanargmin(x)
    Array(4, dtype=int32)

    >>> x = jnp.array([[1, 3, jnp.nan],
    ...                [5, 4, jnp.nan]])
    >>> jnp.nanargmin(x, axis=1)
    Array([0, 1], dtype=int32)

    >>> jnp.nanargmin(x, axis=1, keepdims=True)
    Array([[0],
           [1]], dtype=int32)

[](jax.numpy.nanargmax.html "previous page")

previous

jax.numpy.nanargmax

[](jax.numpy.nancumprod.html "next page")

next

jax.numpy.nancumprod

Contents

- [`nanargmin()`](#jax.numpy.nanargmin)

By The JAX authors

© Copyright 2024, The JAX Authors.\
