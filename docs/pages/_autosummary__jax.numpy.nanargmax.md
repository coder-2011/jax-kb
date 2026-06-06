- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.nanargmax

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.nanargmax.rst "Download source file")
-  .pdf

# jax.numpy.nanargmax

## Contents

- [`nanargmax()`](#jax.numpy.nanargmax)

# jax.numpy.nanargmax[\#](#jax-numpy-nanargmax "Link to this heading")

jax.numpy.nanargmax(*a*, *axis=None*, *out=None*, *keepdims=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L8340-L8396)[\#](#jax.numpy.nanargmax "Link to this definition")  
Return the index of the maximum value of an array, ignoring NaNs.

JAX implementation of [`numpy.nanargmax()`](https://numpy.org/doc/stable/reference/generated/numpy.nanargmax.html#numpy.nanargmax "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional integer specifying the axis along which to find the maximum value. If `axis` is not specified, `a` will be flattened.

- **out** (*None*) – unused by JAX

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None*) – if True, then return an array with the same number of dimensions as `a`.

Returns:  
an array containing the index of the maximum value along the specified axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

In the case of an axis with all-NaN values, the returned index will be -1. This differs from the behavior of [`numpy.nanargmax()`](https://numpy.org/doc/stable/reference/generated/numpy.nanargmax.html#numpy.nanargmax "(in NumPy v2.4)"), which raises an error.

See also

- [`jax.numpy.argmax()`](jax.numpy.argmax.html#jax.numpy.argmax "jax.numpy.argmax"): return the index of the maximum value.

- [`jax.numpy.nanargmin()`](jax.numpy.nanargmin.html#jax.numpy.nanargmin "jax.numpy.nanargmin"): compute `argmin` while ignoring NaN values.

Examples

    >>> x = jnp.array([1, 3, 5, 4, jnp.nan])

Using a standard [`argmax()`](jax.numpy.argmax.html#jax.numpy.argmax "jax.numpy.argmax") leads to potentially unexpected results:

    >>> jnp.argmax(x)
    Array(4, dtype=int32)

Using `nanargmax` returns the index of the maximum non-NaN value.

    >>> jnp.nanargmax(x)
    Array(2, dtype=int32)

    >>> x = jnp.array([[1, 3, jnp.nan],
    ...                [5, 4, jnp.nan]])
    >>> jnp.nanargmax(x, axis=1)
    Array([1, 0], dtype=int32)

    >>> jnp.nanargmax(x, axis=1, keepdims=True)
    Array([[1],
           [0]], dtype=int32)

[](jax.numpy.nan_to_num.html "previous page")

previous

jax.numpy.nan_to_num

[](jax.numpy.nanargmin.html "next page")

next

jax.numpy.nanargmin

Contents

- [`nanargmax()`](#jax.numpy.nanargmax)

By The JAX authors

© Copyright 2024, The JAX Authors.\
