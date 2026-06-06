- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.pad

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.pad.rst "Download source file")
-  .pdf

# jax.numpy.pad

## Contents

- [`pad()`](#jax.numpy.pad)

# jax.numpy.pad[\#](#jax-numpy-pad "Link to this heading")

jax.numpy.pad(*array*, *pad_width*, *mode='constant'*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L4202-L4355)[\#](#jax.numpy.pad "Link to this definition")  
Add padding to an array.

JAX implementation of [`numpy.pad()`](https://numpy.org/doc/stable/reference/generated/numpy.pad.html#numpy.pad "(in NumPy v2.4)").

Parameters:  
- **array** (*ArrayLike*) – array to pad.

- **pad_width** (*PadValueLike\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *np.ndarray\]*) –

  specify the pad width for each dimension of an array. Padding widths may be separately specified for *before* and *after* the array. Options are:

  - `int` or `(int,)`: pad each array dimension with the same number of values both before and after.

  - `(before,`` ``after)`: pad each array with `before` elements before, and `after` elements after

  - `((before_1,`` ``after_1),`` ``(before_2,`` ``after_2),`` ``...`` ``(before_N,`` ``after_N))`: specify distinct `before` and `after` values for each array dimension.

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]*) –

  a string or callable. Supported pad modes are:

  - `'constant'` (default): pad with a constant value, which defaults to zero.

  - `'empty'`: pad with empty values (i.e. zero)

  - `'edge'`: pad with the edge values of the array.

  - `'wrap'`: pad by wrapping the array.

  - `'linear_ramp'`: pad with a linear ramp to specified `end_values`.

  - `'maximum'`: pad with the maximum value.

  - `'mean'`: pad with the mean value.

  - `'median'`: pad with the median value.

  - `'minimum'`: pad with the minimum value.

  - `'reflect'`: pad by reflection.

  - `'symmetric'`: pad by symmetric reflection.

  - `<callable>`: a callable function. See Notes below.

- **constant_values** – referenced for `mode`` ``=`` ``'constant'`. Specify the constant value to pad with.

- **stat_length** – referenced for `mode`` ``in`` ``['maximum',`` ``'mean',`` ``'median',`` ``'minimum']`. An integer or tuple specifying the number of edge values to use when calculating the statistic.

- **end_values** – referenced for `mode`` ``=`` ``'linear_ramp'`. Specify the end values to ramp the padding values to.

- **reflect_type** – referenced for `mode`` ``in`` ``['reflect',`` ``'symmetric']`. Specify whether to use even or odd reflection.

Returns:  
A padded copy of `array`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

When `mode` is callable, it should have the following signature:

    def pad_func(row: Array, pad_width: tuple[int, int],
                 iaxis: int, kwargs: dict) -> Array:
      ...

Here `row` is a 1D slice of the padded array along axis `iaxis`, with the pad values filled with zeros. `pad_width` is a tuple specifying the `(before,`` ``after)` padding sizes, and `kwargs` are any additional keyword arguments passed to the [`jax.numpy.pad()`](#jax.numpy.pad "jax.numpy.pad") function.

Note that while in NumPy, the function should modify `row` in-place, in JAX the function should return the modified `row`. In JAX, the custom padding function will be mapped across the padded axis using the [`jax.vmap()`](jax.vmap.html#jax.vmap "jax.vmap") transformation.

See also

- [`jax.numpy.resize()`](jax.numpy.resize.html#jax.numpy.resize "jax.numpy.resize"): resize an array

- [`jax.numpy.tile()`](jax.numpy.tile.html#jax.numpy.tile "jax.numpy.tile"): create a larger array by tiling a smaller array.

- [`jax.numpy.repeat()`](jax.numpy.repeat.html#jax.numpy.repeat "jax.numpy.repeat"): create a larger array by repeating values of a smaller array.

Examples

Pad a 1-dimensional array with zeros:

    >>> x = jnp.array([10, 20, 30, 40])
    >>> jnp.pad(x, 2)
    Array([ 0,  0, 10, 20, 30, 40,  0,  0], dtype=int32)
    >>> jnp.pad(x, (2, 4))
    Array([ 0,  0, 10, 20, 30, 40,  0,  0,  0,  0], dtype=int32)

Pad a 1-dimensional array with specified values:

    >>> jnp.pad(x, 2, constant_values=99)
    Array([99, 99, 10, 20, 30, 40, 99, 99], dtype=int32)

Pad a 1-dimensional array with the mean array value:

    >>> jnp.pad(x, 2, mode='mean')
    Array([25, 25, 10, 20, 30, 40, 25, 25], dtype=int32)

Pad a 1-dimensional array with reflected values:

    >>> jnp.pad(x, 2, mode='reflect')
    Array([30, 20, 10, 20, 30, 40, 30, 20], dtype=int32)

Pad a 2-dimensional array with different paddings in each dimension:

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> jnp.pad(x, ((1, 2), (3, 0)))
    Array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 2, 3],
           [0, 0, 0, 4, 5, 6],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0]], dtype=int32)

Pad a 1-dimensional array with a custom padding function:

    >>> def custom_pad(row, pad_width, iaxis, kwargs):
    ...   # row represents a 1D slice of the zero-padded array.
    ...   before, after = pad_width
    ...   before_value = kwargs.get('before_value', 0)
    ...   after_value = kwargs.get('after_value', 0)
    ...   row = row.at[:before].set(before_value)
    ...   return row.at[len(row) - after:].set(after_value)
    >>> x = jnp.array([2, 3, 4])
    >>> jnp.pad(x, 2, custom_pad, before_value=-10, after_value=10)
    Array([-10, -10,   2,   3,   4,  10,  10], dtype=int32)

[](jax.numpy.packbits.html "previous page")

previous

jax.numpy.packbits

[](jax.numpy.partition.html "next page")

next

jax.numpy.partition

Contents

- [`pad()`](#jax.numpy.pad)

By The JAX authors

© Copyright 2024, The JAX Authors.\
