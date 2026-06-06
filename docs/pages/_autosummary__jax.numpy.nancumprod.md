- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.nancumprod

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.nancumprod.rst "Download source file")
-  .pdf

# jax.numpy.nancumprod

## Contents

- [`nancumprod()`](#jax.numpy.nancumprod)

# jax.numpy.nancumprod[\#](#jax-numpy-nancumprod "Link to this heading")

jax.numpy.nancumprod(*a*, *axis=None*, *dtype=None*, *out=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L2202-L2249)[\#](#jax.numpy.nancumprod "Link to this definition")  
Cumulative product of elements along an axis, ignoring NaN values.

JAX implementation of [`numpy.nancumprod()`](https://numpy.org/doc/stable/reference/generated/numpy.nancumprod.html#numpy.nancumprod "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – N-dimensional array to be accumulated.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – integer axis along which to accumulate. If None (default), then array will be flattened and accumulated along the flattened axis.

- **dtype** (*DTypeLike* *\|* *None*) – optionally specify the dtype of the output. If not specified, then the output dtype will match the input dtype.

- **out** (*None*) – unused by JAX

Returns:  
An array containing the accumulated product along the given axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.cumprod()`](jax.numpy.cumprod.html#jax.numpy.cumprod "jax.numpy.cumprod"): cumulative product without ignoring NaN values.

- `jax.numpy.multiply.accumulate()`: cumulative product via ufunc methods.

- [`jax.numpy.prod()`](jax.numpy.prod.html#jax.numpy.prod "jax.numpy.prod"): product along axis

Examples

    >>> x = jnp.array([[1., 2., jnp.nan],
    ...                [4., jnp.nan, 6.]])

The standard cumulative product will propagate NaN values:

    >>> jnp.cumprod(x)
    Array([ 1.,  2., nan, nan, nan, nan], dtype=float32)

[`nancumprod()`](#jax.numpy.nancumprod "jax.numpy.nancumprod") will ignore NaN values, effectively replacing them with ones:

    >>> jnp.nancumprod(x)
    Array([ 1.,  2.,  2.,  8.,  8., 48.], dtype=float32)

Cumulative product along axis 1:

    >>> jnp.nancumprod(x, axis=1)
    Array([[ 1.,  2.,  2.],
           [ 4.,  4., 24.]], dtype=float32)

[](jax.numpy.nanargmin.html "previous page")

previous

jax.numpy.nanargmin

[](jax.numpy.nancumsum.html "next page")

next

jax.numpy.nancumsum

Contents

- [`nancumprod()`](#jax.numpy.nancumprod)

By The JAX authors

© Copyright 2024, The JAX Authors.\
