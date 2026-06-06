- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.argmax

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.argmax.rst "Download source file")
-  .pdf

# jax.numpy.argmax

## Contents

- [`argmax()`](#jax.numpy.argmax)

# jax.numpy.argmax[\#](#jax-numpy-argmax "Link to this heading")

jax.numpy.argmax(*a*, *axis=None*, *out=None*, *keepdims=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L8218-L8263)[\#](#jax.numpy.argmax "Link to this definition")  
Return the index of the maximum value of an array.

JAX implementation of [`numpy.argmax()`](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html#numpy.argmax "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional integer specifying the axis along which to find the maximum value. If `axis` is not specified, `a` will be flattened.

- **out** (*None*) – unused by JAX

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None*) – if True, then return an array with the same number of dimensions as `a`.

Returns:  
an array containing the index of the maximum value along the specified axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.argmin()`](jax.numpy.argmin.html#jax.numpy.argmin "jax.numpy.argmin"): return the index of the minimum value.

- [`jax.numpy.nanargmax()`](jax.numpy.nanargmax.html#jax.numpy.nanargmax "jax.numpy.nanargmax"): compute `argmax` while ignoring NaN values.

Note

When the maximum value occurs more than once along a particular axis, the smallest index is returned.

Examples

    >>> x = jnp.array([1, 3, 5, 4, 2])
    >>> jnp.argmax(x)
    Array(2, dtype=int32)

    >>> x = jnp.array([[1, 3, 2],
    ...                [5, 4, 1]])
    >>> jnp.argmax(x, axis=1)
    Array([1, 0], dtype=int32)

    >>> jnp.argmax(x, axis=1, keepdims=True)
    Array([[1],
           [0]], dtype=int32)

[](jax.numpy.arctanh.html "previous page")

previous

jax.numpy.arctanh

[](jax.numpy.argmin.html "next page")

next

jax.numpy.argmin

Contents

- [`argmax()`](#jax.numpy.argmax)

By The JAX authors

© Copyright 2024, The JAX Authors.\
