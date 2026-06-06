- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.cumprod

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.cumprod.rst "Download source file")
-  .pdf

# jax.numpy.cumprod

## Contents

- [`cumprod()`](#jax.numpy.cumprod)

# jax.numpy.cumprod[\#](#jax-numpy-cumprod "Link to this heading")

jax.numpy.cumprod(*a*, *axis=None*, *dtype=None*, *out=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L2116-L2150)[\#](#jax.numpy.cumprod "Link to this definition")  
Cumulative product of elements along an axis.

JAX implementation of [`numpy.cumprod()`](https://numpy.org/doc/stable/reference/generated/numpy.cumprod.html#numpy.cumprod "(in NumPy v2.4)").

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

- `jax.numpy.multiply.accumulate()`: cumulative product via ufunc methods.

- [`jax.numpy.nancumprod()`](jax.numpy.nancumprod.html#jax.numpy.nancumprod "jax.numpy.nancumprod"): cumulative product ignoring NaN values.

- [`jax.numpy.prod()`](jax.numpy.prod.html#jax.numpy.prod "jax.numpy.prod"): product along axis

Examples

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> jnp.cumprod(x)  # flattened cumulative product
    Array([  1,   2,   6,  24, 120, 720], dtype=int32)
    >>> jnp.cumprod(x, axis=1)  # cumulative product along axis 1
    Array([[  1,   2,   6],
           [  4,  20, 120]], dtype=int32)

[](jax.numpy.csingle.html "previous page")

previous

jax.numpy.csingle

[](jax.numpy.cumsum.html "next page")

next

jax.numpy.cumsum

Contents

- [`cumprod()`](#jax.numpy.cumprod)

By The JAX authors

© Copyright 2024, The JAX Authors.\
