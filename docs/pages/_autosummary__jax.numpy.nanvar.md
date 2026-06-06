- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.nanvar

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.nanvar.rst "Download source file")
-  .pdf

# jax.numpy.nanvar

## Contents

- [`nanvar()`](#jax.numpy.nanvar)

# jax.numpy.nanvar[\#](#jax-numpy-nanvar "Link to this heading")

jax.numpy.nanvar(*a*, *axis=None*, *dtype=None*, *out=None*, *ddof=0*, *keepdims=False*, *where=None*, *mean=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L1827-L1916)[\#](#jax.numpy.nanvar "Link to this definition")  
Compute the variance of array elements along a given axis, ignoring NaNs.

JAX implementation of [`numpy.nanvar()`](https://numpy.org/doc/stable/reference/generated/numpy.nanvar.html#numpy.nanvar "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array.

- **axis** (*Axis*) – optional, int or sequence of ints, default=None. Axis along which the variance is computed. If None, variance is computed along flattened array.

- **dtype** (*DTypeLike* *\|* *None*) – The type of the output array. Default=None.

- **ddof** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int, default=0. Degrees of freedom. The divisor in the variance computation is `N-ddof`, `N` is number of elements along given axis.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If true, reduced axes are left in the result with size 1.

- **where** (*ArrayLike* *\|* *None*) – optional, boolean array, default=None. The elements to be used in the variance. Array should be broadcast compatible to the input.

- **mean** (*ArrayLike* *\|* *None*) – optional, mean of the input array, computed along the given axis. If provided, it will be used to compute the variance instead of computing it from the input array. If specified, mean must be broadcast-compatible with the input array. In the general case, this can be achieved by computing the mean with `keepdims=True` and `axis` matching this function’s `axis` argument.

- **out** (*None*) – Unused by JAX.

Returns:  
An array containing the variance of array elements along specified axis. If all elements along the given axis are NaNs, returns `nan`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.nanmean()`](jax.numpy.nanmean.html#jax.numpy.nanmean "jax.numpy.nanmean"): Compute the mean of array elements over a given axis, ignoring NaNs.

- [`jax.numpy.nanstd()`](jax.numpy.nanstd.html#jax.numpy.nanstd "jax.numpy.nanstd"): Computed the standard deviation of a given axis, ignoring NaNs.

- [`jax.numpy.var()`](jax.numpy.var.html#jax.numpy.var "jax.numpy.var"): Compute the variance of array elements along a given axis.

Examples

By default, `jnp.nanvar` computes the variance along all axes.

    >>> nan = jnp.nan
    >>> x = jnp.array([[1, nan, 4, 3],
    ...                [nan, 2, nan, 9],
    ...                [4, 8, 6, nan]])
    >>> jnp.nanvar(x)
    Array(6.984375, dtype=float32)

If `axis=1`, variance is computed along axis 1.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.nanvar(x, axis=1))
    [ 1.56 12.25  2.67]

To preserve the dimensions of input, you can set `keepdims=True`.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.nanvar(x, axis=1, keepdims=True))
    [[ 1.56]
     [12.25]
     [ 2.67]]

If `ddof=1`:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.nanvar(x, axis=1, keepdims=True, ddof=1))
    [[ 2.33]
     [24.5 ]
     [ 4.  ]]

To include specific elements of the array to compute variance, you can use `where`.

    >>> where = jnp.array([[1, 0, 1, 0],
    ...                    [0, 1, 1, 0],
    ...                    [1, 1, 0, 1]], dtype=bool)
    >>> jnp.nanvar(x, axis=1, keepdims=True, where=where)
    Array([[2.25],
           [0.  ],
           [4.  ]], dtype=float32)

[](jax.numpy.nansum.html "previous page")

previous

jax.numpy.nansum

[](jax.numpy.ndarray.html "next page")

next

jax.numpy.ndarray

Contents

- [`nanvar()`](#jax.numpy.nanvar)

By The JAX authors

© Copyright 2024, The JAX Authors.\
