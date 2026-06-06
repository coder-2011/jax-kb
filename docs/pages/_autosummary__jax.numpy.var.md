- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.var

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.var.rst "Download source file")
-  .pdf

# jax.numpy.var

## Contents

- [`var()`](#jax.numpy.var)

# jax.numpy.var[\#](#jax-numpy-var "Link to this heading")

jax.numpy.var(*a*, *axis=None*, *dtype=None*, *out=None*, *ddof=0*, *keepdims=False*, *\**, *where=None*, *mean=None*, *correction=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L1030-L1120)[\#](#jax.numpy.var "Link to this definition")  
Compute the variance along a given axis.

JAX implementation of [`numpy.var()`](https://numpy.org/doc/stable/reference/generated/numpy.var.html#numpy.var "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array.

- **axis** (*Axis*) – optional, int or sequence of ints, default=None. Axis along which the variance is computed. If None, variance is computed along all the axes.

- **dtype** (*DTypeLike* *\|* *None*) – The type of the output array. Default=None.

- **ddof** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int, default=0. Degrees of freedom. The divisor in the variance computation is `N-ddof`, `N` is number of elements along given axis.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If true, reduced axes are left in the result with size 1.

- **where** (*ArrayLike* *\|* *None*) – optional, boolean array, default=None. The elements to be used in the variance. Array should be broadcast compatible to the input.

- **mean** (*ArrayLike* *\|* *None*) – optional, mean of the input array, computed along the given axis. If provided, it will be used to compute the variance instead of computing it from the input array. If specified, mean must be broadcast-compatible with the input array. In the general case, this can be achieved by computing the mean with `keepdims=True` and `axis` matching this function’s `axis` argument.

- **correction** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* *None*) – int or float, default=None. Alternative name for `ddof`. Both ddof and correction can’t be provided simultaneously.

- **out** (*None*) – Unused by JAX.

Returns:  
An array of the variance along the given axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.mean()`](jax.numpy.mean.html#jax.numpy.mean "jax.numpy.mean"): Compute the mean of array elements over a given axis.

- [`jax.numpy.std()`](jax.numpy.std.html#jax.numpy.std "jax.numpy.std"): Compute the standard deviation of array elements over given axis.

- [`jax.numpy.nanvar()`](jax.numpy.nanvar.html#jax.numpy.nanvar "jax.numpy.nanvar"): Compute the variance along a given axis, ignoring NaNs values.

- [`jax.numpy.nanstd()`](jax.numpy.nanstd.html#jax.numpy.nanstd "jax.numpy.nanstd"): Computed the standard deviation of a given axis, ignoring NaN values.

Examples

By default, `jnp.var` computes the variance along all axes.

    >>> x = jnp.array([[1, 3, 4, 2],
    ...                [5, 2, 6, 3],
    ...                [8, 4, 2, 9]])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.var(x)
    Array(5.74, dtype=float32)

If `axis=1`, variance is computed along axis 1.

    >>> jnp.var(x, axis=1)
    Array([1.25  , 2.5   , 8.1875], dtype=float32)

To preserve the dimensions of input, you can set `keepdims=True`.

    >>> jnp.var(x, axis=1, keepdims=True)
    Array([[1.25  ],
           [2.5   ],
           [8.1875]], dtype=float32)

If `ddof=1`:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.var(x, axis=1, keepdims=True, ddof=1))
    [[ 1.67]
     [ 3.33]
     [10.92]]

To include specific elements of the array to compute variance, you can use `where`.

    >>> where = jnp.array([[1, 0, 1, 0],
    ...                    [0, 1, 1, 0],
    ...                    [1, 1, 1, 0]], dtype=bool)
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.var(x, axis=1, keepdims=True, where=where))
    [[2.25]
     [4.  ]
     [6.22]]

[](jax.numpy.vander.html "previous page")

previous

jax.numpy.vander

[](jax.numpy.vdot.html "next page")

next

jax.numpy.vdot

Contents

- [`var()`](#jax.numpy.var)

By The JAX authors

© Copyright 2024, The JAX Authors.\
