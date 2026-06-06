- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.argmin

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.argmin.rst "Download source file")
-  .pdf

# jax.numpy.argmin

## Contents

- [`argmin()`](#jax.numpy.argmin)

# jax.numpy.argmin[\#](#jax-numpy-argmin "Link to this heading")

jax.numpy.argmin(*a*, *axis=None*, *out=None*, *keepdims=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L8279-L8324)[\#](#jax.numpy.argmin "Link to this definition")  
Return the index of the minimum value of an array.

JAX implementation of [`numpy.argmin()`](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html#numpy.argmin "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional integer specifying the axis along which to find the minimum value. If `axis` is not specified, `a` will be flattened.

- **out** (*None*) – unused by JAX

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None*) – if True, then return an array with the same number of dimensions as `a`.

Returns:  
an array containing the index of the minimum value along the specified axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

When the minimum value occurs more than once along a particular axis, the smallest index is returned.

See also

- [`jax.numpy.argmax()`](jax.numpy.argmax.html#jax.numpy.argmax "jax.numpy.argmax"): return the index of the maximum value.

- [`jax.numpy.nanargmin()`](jax.numpy.nanargmin.html#jax.numpy.nanargmin "jax.numpy.nanargmin"): compute `argmin` while ignoring NaN values.

Examples

    >>> x = jnp.array([1, 3, 5, 4, 2])
    >>> jnp.argmin(x)
    Array(0, dtype=int32)

    >>> x = jnp.array([[1, 3, 2],
    ...                [5, 4, 1]])
    >>> jnp.argmin(x, axis=1)
    Array([0, 2], dtype=int32)

    >>> jnp.argmin(x, axis=1, keepdims=True)
    Array([[0],
           [2]], dtype=int32)

[](jax.numpy.argmax.html "previous page")

previous

jax.numpy.argmax

[](jax.numpy.argpartition.html "next page")

next

jax.numpy.argpartition

Contents

- [`argmin()`](#jax.numpy.argmin)

By The JAX authors

© Copyright 2024, The JAX Authors.\
