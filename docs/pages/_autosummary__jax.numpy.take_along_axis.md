- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.take_along_axis

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.take_along_axis.rst "Download source file")
-  .pdf

# jax.numpy.take_along_axis

## Contents

- [`take_along_axis()`](#jax.numpy.take_along_axis)

# jax.numpy.take_along_axis[\#](#jax-numpy-take-along-axis "Link to this heading")

jax.numpy.take_along_axis(*arr*, *indices*, *axis=-1*, *mode=None*, *fill_value=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/indexing.py#L758-L949)[\#](#jax.numpy.take_along_axis "Link to this definition")  
Take elements from an array.

JAX implementation of [`numpy.take_along_axis()`](https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html#numpy.take_along_axis "(in NumPy v2.4)"), implemented in terms of [`jax.lax.gather()`](jax.lax.gather.html#jax.lax.gather "jax.lax.gather"). JAX’s behavior differs from NumPy in the case of out-of-bound indices; see the `mode` parameter below.

Parameters:  
- **a** – array from which to take values.

- **indices** (*ArrayLike*) – array of integer indices. If `axis` is `None`, must be one-dimensional. If `axis` is not None, must have `a.ndim`` ``==`` ``indices.ndim`, and `a` must be broadcast-compatible with `indices` along dimensions other than `axis`.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – the axis along which to take values. If not specified, the array will be flattened before indexing is applied.

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*slicing.GatherScatterMode*](../jax.lax.html#jax.lax.GatherScatterMode "jax._src.lax.slicing.GatherScatterMode") *\|* *None*) – Out-of-bounds indexing mode, either `"fill"` or `"clip"`. The default `mode="fill"` returns invalid values (e.g. NaN) for out-of bounds indices. For more discussion of `mode` options, see [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at").

- **arr** (*ArrayLike*)

- **fill_value** (*StaticScalar* *\|* *None*)

Returns:  
Array of values extracted from `a`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at"): take values via indexing syntax.

- [`jax.numpy.take()`](jax.numpy.take.html#jax.numpy.take "jax.numpy.take"): take the same indices along every axis slice.

Examples

    >>> x = jnp.array([[1., 2., 3.],
    ...                [4., 5., 6.]])
    >>> indices = jnp.array([[0, 2],
    ...                      [1, 0]])
    >>> jnp.take_along_axis(x, indices, axis=1)
    Array([[1., 3.],
           [5., 4.]], dtype=float32)
    >>> x[jnp.arange(2)[:, None], indices]  # equivalent via indexing syntax
    Array([[1., 3.],
           [5., 4.]], dtype=float32)

Out-of-bound indices fill with invalid values. For float inputs, this is NaN:

    >>> indices = jnp.array([[1, 0, 2]])
    >>> jnp.take_along_axis(x, indices, axis=0)
    Array([[ 4.,  2., nan]], dtype=float32)
    >>> x.at[indices, jnp.arange(3)].get(
    ...     mode='fill', fill_value=jnp.nan)  # equivalent via indexing syntax
    Array([[ 4.,  2., nan]], dtype=float32)

`take_along_axis` is helpful for extracting values from multi-dimensional argsorts and arg reductions. For, here we compute [`argsort()`](jax.numpy.argsort.html#jax.numpy.argsort "jax.numpy.argsort") indices along an axis, and use `take_along_axis` to construct the sorted array:

    >>> x = jnp.array([[5, 3, 4],
    ...                [2, 7, 6]])
    >>> indices = jnp.argsort(x, axis=1)
    >>> indices
    Array([[1, 2, 0],
           [0, 2, 1]], dtype=int32)
    >>> jnp.take_along_axis(x, indices, axis=1)
    Array([[3, 4, 5],
           [2, 6, 7]], dtype=int32)

Similarly, we can use [`argmin()`](jax.numpy.argmin.html#jax.numpy.argmin "jax.numpy.argmin") with `keepdims=True` and use `take_along_axis` to extract the minimum value:

    >>> idx = jnp.argmin(x, axis=1, keepdims=True)
    >>> idx
    Array([[1],
           [0]], dtype=int32)
    >>> jnp.take_along_axis(x, idx, axis=1)
    Array([[3],
           [2]], dtype=int32)

[](jax.numpy.take.html "previous page")

previous

jax.numpy.take

[](jax.numpy.tan.html "next page")

next

jax.numpy.tan

Contents

- [`take_along_axis()`](#jax.numpy.take_along_axis)

By The JAX authors

© Copyright 2024, The JAX Authors.\
