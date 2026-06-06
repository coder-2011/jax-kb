- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.cumulative_sum

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.cumulative_sum.rst "Download source file")
-  .pdf

# jax.numpy.cumulative_sum

## Contents

- [`cumulative_sum()`](#jax.numpy.cumulative_sum)

# jax.numpy.cumulative_sum[\#](#jax-numpy-cumulative-sum "Link to this heading")

jax.numpy.cumulative_sum(*x*, */*, *\**, *axis=None*, *dtype=None*, *include_initial=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L2259-L2319)[\#](#jax.numpy.cumulative_sum "Link to this definition")  
Cumulative sum along the axis of an array.

JAX implementation of [`numpy.cumulative_sum()`](https://numpy.org/doc/stable/reference/generated/numpy.cumulative_sum.html#numpy.cumulative_sum "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – N-dimensional array

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – integer axis along which to accumulate. If `x` is one-dimensional, this argument is optional and defaults to zero.

- **dtype** (*DTypeLike* *\|* *None*) – optional dtype of the output.

- **include_initial** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, then include the initial value in the cumulative sum. Default is False.

Returns:  
An array containing the accumulated values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.cumsum()`](jax.numpy.cumsum.html#jax.numpy.cumsum "jax.numpy.cumsum"): alternative API for cumulative sum.

- [`jax.numpy.nancumsum()`](jax.numpy.nancumsum.html#jax.numpy.nancumsum "jax.numpy.nancumsum"): cumulative sum while ignoring NaN values.

- `jax.numpy.add.accumulate()`: cumulative sum via the ufunc API.

Examples

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> jnp.cumulative_sum(x, axis=1)
    Array([[ 1,  3,  6],
           [ 4,  9, 15]], dtype=int32)
    >>> jnp.cumulative_sum(x, axis=1, include_initial=True)
    Array([[ 0,  1,  3,  6],
           [ 0,  4,  9, 15]], dtype=int32)

[](jax.numpy.cumulative_prod.html "previous page")

previous

jax.numpy.cumulative_prod

[](jax.numpy.deg2rad.html "next page")

next

jax.numpy.deg2rad

Contents

- [`cumulative_sum()`](#jax.numpy.cumulative_sum)

By The JAX authors

© Copyright 2024, The JAX Authors.\
