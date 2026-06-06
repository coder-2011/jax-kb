- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.apply_along_axis

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.apply_along_axis.rst "Download source file")
-  .pdf

# jax.numpy.apply_along_axis

## Contents

- [`apply_along_axis()`](#jax.numpy.apply_along_axis)

# jax.numpy.apply_along_axis[\#](#jax-numpy-apply-along-axis "Link to this heading")

jax.numpy.apply_along_axis(*func1d*, *axis*, *arr*, *\*args*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L7807-L7883)[\#](#jax.numpy.apply_along_axis "Link to this definition")  
Apply a function to 1D array slices along an axis.

JAX implementation of [`numpy.apply_along_axis()`](https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html#numpy.apply_along_axis "(in NumPy v2.4)"). While NumPy implements this iteratively, JAX implements this via [`jax.vmap()`](jax.vmap.html#jax.vmap "jax.vmap"), and so `func1d` must be compatible with `vmap`.

Parameters:  
- **func1d** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")) – a callable function with signature `func1d(arr,`` ``/,`` ``*args,`` ``**kwargs)` where `*args` and `**kwargs` are the additional positional and keyword arguments passed to [`apply_along_axis()`](#jax.numpy.apply_along_axis "jax.numpy.apply_along_axis").

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – integer axis along which to apply the function.

- **arr** (*ArrayLike*) – the array over which to apply the function.

- **args** – additional positional and keyword arguments are passed through to `func1d`.

- **kwargs** – additional positional and keyword arguments are passed through to `func1d`.

Returns:  
The result of `func1d` applied along the specified axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.vmap()`](jax.vmap.html#jax.vmap "jax.vmap"): a more direct way to create a vectorized version of a function.

- [`jax.numpy.apply_over_axes()`](jax.numpy.apply_over_axes.html#jax.numpy.apply_over_axes "jax.numpy.apply_over_axes"): repeatedly apply a function over multiple axes.

- [`jax.numpy.vectorize()`](jax.numpy.vectorize.html#jax.numpy.vectorize "jax.numpy.vectorize"): create a vectorized version of a function.

Examples

A simple example in two dimensions, where the function is applied either row-wise or column-wise:

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> def func1d(x):
    ...   return jnp.sum(x ** 2)
    >>> jnp.apply_along_axis(func1d, 0, x)
    Array([17, 29, 45], dtype=int32)
    >>> jnp.apply_along_axis(func1d, 1, x)
    Array([14, 77], dtype=int32)

For 2D inputs, this can be equivalently expressed using [`jax.vmap()`](jax.vmap.html#jax.vmap "jax.vmap"), though note that vmap specifies the mapped axis rather than the applied axis:

    >>> jax.vmap(func1d, in_axes=1)(x)  # same as applying along axis 0
    Array([17, 29, 45], dtype=int32)
    >>> jax.vmap(func1d, in_axes=0)(x)  # same as applying along axis 1
    Array([14, 77], dtype=int32)

For 3D inputs, [`apply_along_axis()`](#jax.numpy.apply_along_axis "jax.numpy.apply_along_axis") is equivalent to mapping over two dimensions:

    >>> x_3d = jnp.arange(24).reshape(2, 3, 4)
    >>> jnp.apply_along_axis(func1d, 2, x_3d)
    Array([[  14,  126,  366],
           [ 734, 1230, 1854]], dtype=int32)
    >>> jax.vmap(jax.vmap(func1d))(x_3d)
    Array([[  14,  126,  366],
           [ 734, 1230, 1854]], dtype=int32)

The applied function may also take arbitrary positional or keyword arguments, which should be passed directly as additional arguments to [`apply_along_axis()`](#jax.numpy.apply_along_axis "jax.numpy.apply_along_axis"):

    >>> def func1d(x, exponent):
    ...   return jnp.sum(x ** exponent)
    >>> jnp.apply_along_axis(func1d, 0, x, exponent=3)
    Array([ 65, 133, 243], dtype=int32)

[](jax.numpy.append.html "previous page")

previous

jax.numpy.append

[](jax.numpy.apply_over_axes.html "next page")

next

jax.numpy.apply_over_axes

Contents

- [`apply_along_axis()`](#jax.numpy.apply_along_axis)

By The JAX authors

© Copyright 2024, The JAX Authors.\
