- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.std

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.std.rst "Download source file")
-  .pdf

# jax.numpy.std

## Contents

- [`std()`](#jax.numpy.std)

# jax.numpy.std[\#](#jax-numpy-std "Link to this heading")

jax.numpy.std(*a*, *axis=None*, *dtype=None*, *out=None*, *ddof=0*, *keepdims=False*, *\**, *where=None*, *mean=None*, *correction=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L1182-L1268)[\#](#jax.numpy.std "Link to this definition")  
Compute the standard deviation along a given axis.

JAX implementation of [`numpy.std()`](https://numpy.org/doc/stable/reference/generated/numpy.std.html#numpy.std "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array.

- **axis** (*Axis*) – optional, int or sequence of ints, default=None. Axis along which the standard deviation is computed. If None, standard deviaiton is computed along all the axes.

- **dtype** (*DTypeLike* *\|* *None*) – The type of the output array. Default=None.

- **ddof** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int, default=0. Degrees of freedom. The divisor in the standard deviation computation is `N-ddof`, `N` is number of elements along given axis.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If true, reduced axes are left in the result with size 1.

- **where** (*ArrayLike* *\|* *None*) – optional, boolean array, default=None. The elements to be used in the standard deviation. Array should be broadcast compatible to the input.

- **mean** (*ArrayLike* *\|* *None*) – optional, mean of the input array, computed along the given axis. If provided, it will be used to compute the standard deviation instead of computing it from the input array. If specified, mean must be broadcast-compatible with the input array. In the general case, this can be achieved by computing the mean with `keepdims=True` and `axis` matching this function’s `axis` argument.

- **correction** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* *None*) – int or float, default=None. Alternative name for `ddof`. Both ddof and correction can’t be provided simultaneously.

- **out** (*None*) – Unused by JAX.

Returns:  
An array of the standard deviation along the given axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.var()`](jax.numpy.var.html#jax.numpy.var "jax.numpy.var"): Compute the variance of array elements over given axis.

- [`jax.numpy.mean()`](jax.numpy.mean.html#jax.numpy.mean "jax.numpy.mean"): Compute the mean of array elements over a given axis.

- [`jax.numpy.nanvar()`](jax.numpy.nanvar.html#jax.numpy.nanvar "jax.numpy.nanvar"): Compute the variance along a given axis, ignoring NaNs values.

- [`jax.numpy.nanstd()`](jax.numpy.nanstd.html#jax.numpy.nanstd "jax.numpy.nanstd"): Computed the standard deviation of a given axis, ignoring NaN values.

Examples

By default, `jnp.std` computes the standard deviation along all axes.

    >>> x = jnp.array([[1, 3, 4, 2],
    ...                [4, 2, 5, 3],
    ...                [5, 4, 2, 3]])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.std(x)
    Array(1.21, dtype=float32)

If `axis=0`, computes along axis 0.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.std(x, axis=0))
    [1.7  0.82 1.25 0.47]

To preserve the dimensions of input, you can set `keepdims=True`.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.std(x, axis=0, keepdims=True))
    [[1.7  0.82 1.25 0.47]]

If `ddof=1`:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.std(x, axis=0, keepdims=True, ddof=1))
    [[2.08 1.   1.53 0.58]]

To include specific elements of the array to compute standard deviation, you can use `where`.

    >>> where = jnp.array([[1, 0, 1, 0],
    ...                    [0, 1, 0, 1],
    ...                    [1, 1, 1, 0]], dtype=bool)
    >>> jnp.std(x, axis=0, keepdims=True, where=where)
    Array([[2., 1., 1., 0.]], dtype=float32)

[](jax.numpy.stack.html "previous page")

previous

jax.numpy.stack

[](jax.numpy.subtract.html "next page")

next

jax.numpy.subtract

Contents

- [`std()`](#jax.numpy.std)

By The JAX authors

© Copyright 2024, The JAX Authors.\
