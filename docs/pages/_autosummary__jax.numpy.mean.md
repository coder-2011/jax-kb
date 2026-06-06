- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.mean

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.mean.rst "Download source file")
-  .pdf

# jax.numpy.mean

## Contents

- [`mean()`](#jax.numpy.mean)

# jax.numpy.mean[\#](#jax-numpy-mean "Link to this heading")

jax.numpy.mean(*a*, *axis=None*, *dtype=None*, *out=None*, *keepdims=False*, *\**, *where=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L803-L869)[\#](#jax.numpy.mean "Link to this definition")  
Return the mean of array elements along a given axis.

JAX implementation of [`numpy.mean()`](https://numpy.org/doc/stable/reference/generated/numpy.mean.html#numpy.mean "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array.

- **axis** (*Axis*) – optional, int or sequence of ints, default=None. Axis along which the mean to be computed. If None, mean is computed along all the axes.

- **dtype** (*DTypeLike* *\|* *None*) – The type of the output array. If None (default) then the output dtype will be match the input dtype for floating point inputs, or be set to float32 or float64 for non-floating-point inputs.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If true, reduced axes are left in the result with size 1.

- **where** (*ArrayLike* *\|* *None*) – optional, boolean array, default=None. The elements to be used in the mean. Array should be broadcast compatible to the input.

- **out** (*None*) – Unused by JAX.

Returns:  
An array of the mean along the given axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

For inputs of type float16 or bfloat16, the reductions will be performed at float32 precision.

See also

- [`jax.numpy.average()`](jax.numpy.average.html#jax.numpy.average "jax.numpy.average"): Compute the weighted average of array elements

- [`jax.numpy.sum()`](jax.numpy.sum.html#jax.numpy.sum "jax.numpy.sum"): Compute the sum of array elements.

Examples

By default, the mean is computed along all the axes.

    >>> x = jnp.array([[1, 3, 4, 2],
    ...                [5, 2, 6, 3],
    ...                [8, 1, 2, 9]])
    >>> jnp.mean(x)
    Array(3.8333335, dtype=float32)

If `axis=1`, the mean is computed along axis 1.

    >>> jnp.mean(x, axis=1)
    Array([2.5, 4. , 5. ], dtype=float32)

If `keepdims=True`, `ndim` of the output is equal to that of the input.

    >>> jnp.mean(x, axis=1, keepdims=True)
    Array([[2.5],
           [4. ],
           [5. ]], dtype=float32)

To use only specific elements of `x` to compute the mean, you can use `where`.

    >>> where = jnp.array([[1, 0, 1, 0],
    ...                    [0, 1, 0, 1],
    ...                    [1, 1, 0, 1]], dtype=bool)
    >>> jnp.mean(x, axis=1, keepdims=True, where=where)
    Array([[2.5],
           [2.5],
           [6. ]], dtype=float32)

[](jax.numpy.maximum.html "previous page")

previous

jax.numpy.maximum

[](jax.numpy.median.html "next page")

next

jax.numpy.median

Contents

- [`mean()`](#jax.numpy.mean)

By The JAX authors

© Copyright 2024, The JAX Authors.\
