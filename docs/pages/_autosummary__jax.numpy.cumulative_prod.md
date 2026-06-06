- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.cumulative_prod

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.cumulative_prod.rst "Download source file")
-  .pdf

# jax.numpy.cumulative_prod

## Contents

- [`cumulative_prod()`](#jax.numpy.cumulative_prod)

# jax.numpy.cumulative_prod[\#](#jax-numpy-cumulative-prod "Link to this heading")

jax.numpy.cumulative_prod(*x*, */*, *\**, *axis=None*, *dtype=None*, *include_initial=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L2321-L2381)[\#](#jax.numpy.cumulative_prod "Link to this definition")  
Cumulative product along the axis of an array.

JAX implementation of [`numpy.cumulative_prod()`](https://numpy.org/doc/stable/reference/generated/numpy.cumulative_prod.html#numpy.cumulative_prod "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – N-dimensional array

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – integer axis along which to accumulate. If `x` is one-dimensional, this argument is optional and defaults to zero.

- **dtype** (*DTypeLike* *\|* *None*) – optional dtype of the output.

- **include_initial** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, then include the initial value in the cumulative product. Default is False.

Returns:  
An array containing the accumulated values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.cumprod()`](jax.numpy.cumprod.html#jax.numpy.cumprod "jax.numpy.cumprod"): alternative API for cumulative product.

- [`jax.numpy.nancumprod()`](jax.numpy.nancumprod.html#jax.numpy.nancumprod "jax.numpy.nancumprod"): cumulative product while ignoring NaN values.

- `jax.numpy.multiply.accumulate()`: cumulative product via the ufunc API.

Examples

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> jnp.cumulative_prod(x, axis=1)
    Array([[  1,   2,   6],
           [  4,  20, 120]], dtype=int32)
    >>> jnp.cumulative_prod(x, axis=1, include_initial=True)
    Array([[  1,   1,   2,   6],
           [  1,   4,  20, 120]], dtype=int32)

[](jax.numpy.cumsum.html "previous page")

previous

jax.numpy.cumsum

[](jax.numpy.cumulative_sum.html "next page")

next

jax.numpy.cumulative_sum

Contents

- [`cumulative_prod()`](#jax.numpy.cumulative_prod)

By The JAX authors

© Copyright 2024, The JAX Authors.\
