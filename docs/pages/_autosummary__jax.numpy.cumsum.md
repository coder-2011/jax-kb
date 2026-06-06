- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.cumsum

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.cumsum.rst "Download source file")
-  .pdf

# jax.numpy.cumsum

## Contents

- [`cumsum()`](#jax.numpy.cumsum)

# jax.numpy.cumsum[\#](#jax-numpy-cumsum "Link to this heading")

jax.numpy.cumsum(*a*, *axis=None*, *dtype=None*, *out=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L2079-L2114)[\#](#jax.numpy.cumsum "Link to this definition")  
Cumulative sum of elements along an axis.

JAX implementation of [`numpy.cumsum()`](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html#numpy.cumsum "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – N-dimensional array to be accumulated.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – integer axis along which to accumulate. If None (default), then array will be flattened and accumulated along the flattened axis.

- **dtype** (*DTypeLike* *\|* *None*) – optionally specify the dtype of the output. If not specified, then the output dtype will match the input dtype.

- **out** (*None*) – unused by JAX

Returns:  
An array containing the accumulated sum along the given axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.cumulative_sum()`](jax.numpy.cumulative_sum.html#jax.numpy.cumulative_sum "jax.numpy.cumulative_sum"): cumulative sum via the array API standard.

- `jax.numpy.add.accumulate()`: cumulative sum via ufunc methods.

- [`jax.numpy.nancumsum()`](jax.numpy.nancumsum.html#jax.numpy.nancumsum "jax.numpy.nancumsum"): cumulative sum ignoring NaN values.

- [`jax.numpy.sum()`](jax.numpy.sum.html#jax.numpy.sum "jax.numpy.sum"): sum along axis

Examples

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> jnp.cumsum(x)  # flattened cumulative sum
    Array([ 1,  3,  6, 10, 15, 21], dtype=int32)
    >>> jnp.cumsum(x, axis=1)  # cumulative sum along axis 1
    Array([[ 1,  3,  6],
           [ 4,  9, 15]], dtype=int32)

[](jax.numpy.cumprod.html "previous page")

previous

jax.numpy.cumprod

[](jax.numpy.cumulative_prod.html "next page")

next

jax.numpy.cumulative_prod

Contents

- [`cumsum()`](#jax.numpy.cumsum)

By The JAX authors

© Copyright 2024, The JAX Authors.\
