- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.take

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.take.rst "Download source file")
-  .pdf

# jax.numpy.take

## Contents

- [`take()`](#jax.numpy.take)

# jax.numpy.take[\#](#jax-numpy-take "Link to this heading")

jax.numpy.take(*a*, *indices*, *axis=None*, *out=None*, *mode=None*, *unique_indices=False*, *indices_are_sorted=False*, *fill_value=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/indexing.py#L600-L688)[\#](#jax.numpy.take "Link to this definition")  
Take elements from an array.

JAX implementation of [`numpy.take()`](https://numpy.org/doc/stable/reference/generated/numpy.take.html#numpy.take "(in NumPy v2.4)"), implemented in terms of [`jax.lax.gather()`](jax.lax.gather.html#jax.lax.gather "jax.lax.gather"). JAX’s behavior differs from NumPy in the case of out-of-bound indices; see the `mode` parameter below.

Parameters:  
- **a** (*ArrayLike*) – array from which to take values.

- **indices** (*ArrayLike*) – N-dimensional array of integer indices of values to take from the array.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – the axis along which to take values. If not specified, the array will be flattened before indexing is applied.

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – Out-of-bounds indexing mode, either `"fill"` or `"clip"`. The default `mode="fill"` returns invalid values (e.g. NaN) for out-of bounds indices; the `fill_value` argument gives control over this value. For more discussion of `mode` options, see [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at").

- **fill_value** (*StaticScalar* *\|* *None*) – The fill value to return for out-of-bounds slices when mode is ‘fill’. Ignored otherwise. Defaults to NaN for inexact types, the largest negative value for signed types, the largest positive value for unsigned types, and True for booleans.

- **unique_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, the implementation will assume that the indices are unique after normalization of negative indices, which lets the compiler emit more efficient code during the backward pass. If set to True and normalized indices are not unique, the result is implementation-defined and may be non-deterministic.

- **indices_are_sorted** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, the implementation will assume that the indices are sorted in ascending order after normalization of negative indices, which can lead to more efficient execution on some backends. If set to True and normalized indices are not sorted, the output is implementation-defined.

- **out** (*None*)

Returns:  
Array of values extracted from `a`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at"): take values via indexing syntax.

- [`jax.numpy.take_along_axis()`](jax.numpy.take_along_axis.html#jax.numpy.take_along_axis "jax.numpy.take_along_axis"): take values along an axis

Examples

    >>> x = jnp.array([[1., 2., 3.],
    ...                [4., 5., 6.]])
    >>> indices = jnp.array([2, 0])

Passing no axis results in indexing into the flattened array:

    >>> jnp.take(x, indices)
    Array([3., 1.], dtype=float32)
    >>> x.ravel()[indices]  # equivalent indexing syntax
    Array([3., 1.], dtype=float32)

Passing an axis results ind applying the index to every subarray along the axis:

    >>> jnp.take(x, indices, axis=1)
    Array([[3., 1.],
           [6., 4.]], dtype=float32)
    >>> x[:, indices]  # equivalent indexing syntax
    Array([[3., 1.],
           [6., 4.]], dtype=float32)

Out-of-bound indices fill with invalid values. For float inputs, this is NaN:

    >>> jnp.take(x, indices, axis=0)
    Array([[nan, nan, nan],
           [ 1.,  2.,  3.]], dtype=float32)
    >>> x.at[indices].get(mode='fill', fill_value=jnp.nan)  # equivalent indexing syntax
    Array([[nan, nan, nan],
           [ 1.,  2.,  3.]], dtype=float32)

This default out-of-bound behavior can be adjusted using the `mode` parameter, for example, we can instead clip to the last valid value:

    >>> jnp.take(x, indices, axis=0, mode='clip')
    Array([[4., 5., 6.],
           [1., 2., 3.]], dtype=float32)
    >>> x.at[indices].get(mode='clip')  # equivalent indexing syntax
    Array([[4., 5., 6.],
           [1., 2., 3.]], dtype=float32)

[](jax.numpy.swapaxes.html "previous page")

previous

jax.numpy.swapaxes

[](jax.numpy.take_along_axis.html "next page")

next

jax.numpy.take_along_axis

Contents

- [`take()`](#jax.numpy.take)

By The JAX authors

© Copyright 2024, The JAX Authors.\
