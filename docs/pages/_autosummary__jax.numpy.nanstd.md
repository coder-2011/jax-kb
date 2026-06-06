- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.nanstd

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.nanstd.rst "Download source file")
-  .pdf

# jax.numpy.nanstd

## Contents

- [`nanstd()`](#jax.numpy.nanstd)

# jax.numpy.nanstd[\#](#jax-numpy-nanstd "Link to this heading")

jax.numpy.nanstd(*a*, *axis=None*, *dtype=None*, *out=None*, *ddof=0*, *keepdims=False*, *where=None*, *mean=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L1944-L2026)[\#](#jax.numpy.nanstd "Link to this definition")  
Compute the standard deviation along a given axis, ignoring NaNs.

JAX implementation of [`numpy.nanstd()`](https://numpy.org/doc/stable/reference/generated/numpy.nanstd.html#numpy.nanstd "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array.

- **axis** (*Axis*) – optional, int or sequence of ints, default=None. Axis along which the standard deviation is computed. If None, standard deviaiton is computed along flattened array.

- **dtype** (*DTypeLike* *\|* *None*) – The type of the output array. Default=None.

- **ddof** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int, default=0. Degrees of freedom. The divisor in the standard deviation computation is `N-ddof`, `N` is number of elements along given axis.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If true, reduced axes are left in the result with size 1.

- **where** (*ArrayLike* *\|* *None*) – optional, boolean array, default=None. The elements to be used in the standard deviation. Array should be broadcast compatible to the input.

- **mean** (*ArrayLike* *\|* *None*) – optional, mean of the input array, computed along the given axis. If provided, it will be used to compute the standard deviation instead of computing it from the input array. If specified, mean must be broadcast-compatible with the input array. In the general case, this can be achieved by computing the mean with `keepdims=True` and `axis` matching this function’s `axis` argument.

- **out** (*None*) – Unused by JAX.

Returns:  
An array containing the standard deviation of array elements along the given axis. If all elements along the given axis are NaNs, returns `nan`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.nanmean()`](jax.numpy.nanmean.html#jax.numpy.nanmean "jax.numpy.nanmean"): Compute the mean of array elements over a given axis, ignoring NaNs.

- [`jax.numpy.nanvar()`](jax.numpy.nanvar.html#jax.numpy.nanvar "jax.numpy.nanvar"): Compute the variance along the given axis, ignoring NaNs values.

- [`jax.numpy.std()`](jax.numpy.std.html#jax.numpy.std "jax.numpy.std"): Computed the standard deviation along the given axis.

Examples

By default, `jnp.nanstd` computes the standard deviation along flattened array.

    >>> nan = jnp.nan
    >>> x = jnp.array([[3, nan, 4, 5],
    ...                [nan, 2, nan, 7],
    ...                [2, 1, 6, nan]])
    >>> jnp.nanstd(x)
    Array(1.9843135, dtype=float32)

If `axis=0`, computes standard deviation along axis 0.

    >>> jnp.nanstd(x, axis=0)
    Array([0.5, 0.5, 1. , 1. ], dtype=float32)

To preserve the dimensions of input, you can set `keepdims=True`.

    >>> jnp.nanstd(x, axis=0, keepdims=True)
    Array([[0.5, 0.5, 1. , 1. ]], dtype=float32)

If `ddof=1`:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.nanstd(x, axis=0, keepdims=True, ddof=1))
    [[0.71 0.71 1.41 1.41]]

To include specific elements of the array to compute standard deviation, you can use `where`.

    >>> where=jnp.array([[1, 0, 1, 0],
    ...                  [0, 1, 0, 1],
    ...                  [1, 1, 0, 1]], dtype=bool)
    >>> jnp.nanstd(x, axis=0, keepdims=True, where=where)
    Array([[0.5, 0.5, 0. , 0. ]], dtype=float32)

[](jax.numpy.nanquantile.html "previous page")

previous

jax.numpy.nanquantile

[](jax.numpy.nansum.html "next page")

next

jax.numpy.nansum

Contents

- [`nanstd()`](#jax.numpy.nanstd)

By The JAX authors

© Copyright 2024, The JAX Authors.\
