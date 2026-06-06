- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.max

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.max.rst "Download source file")
-  .pdf

# jax.numpy.max

## Contents

- [`max()`](#jax.numpy.max)

# jax.numpy.max[\#](#jax-numpy-max "Link to this heading")

jax.numpy.max(*a*, *axis=None*, *out=None*, *keepdims=False*, *initial=None*, *where=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L408-L481)[\#](#jax.numpy.max "Link to this definition")  
Return the maximum of the array elements along a given axis.

JAX implementation of [`numpy.max()`](https://numpy.org/doc/stable/reference/generated/numpy.max.html#numpy.max "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – Input array.

- **axis** (*Axis*) – int or array, default=None. Axis along which the maximum to be computed. If None, the maximum is computed along all the axes.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If true, reduced axes are left in the result with size 1.

- **initial** (*ArrayLike* *\|* *None*) – int or array, default=None. Initial value for the maximum.

- **where** (*ArrayLike* *\|* *None*) – int or array of boolean dtype, default=None. The elements to be used in the maximum. Array should be broadcast compatible to the input. `initial` must be specified when `where` is used.

- **out** (*None*) – Unused by JAX.

Returns:  
An array of maximum values along the given axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.min()`](jax.numpy.min.html#jax.numpy.min "jax.numpy.min"): Compute the minimum of array elements along a given axis.

- [`jax.numpy.sum()`](jax.numpy.sum.html#jax.numpy.sum "jax.numpy.sum"): Compute the sum of array elements along a given axis.

- [`jax.numpy.prod()`](jax.numpy.prod.html#jax.numpy.prod "jax.numpy.prod"): Compute the product of array elements along a given axis.

Examples

By default, `jnp.max` computes the maximum of elements along all the axes.

    >>> x = jnp.array([[9, 3, 4, 5],
    ...                [5, 2, 7, 4],
    ...                [8, 1, 3, 6]])
    >>> jnp.max(x)
    Array(9, dtype=int32)

If `axis=1`, the maximum will be computed along axis 1.

    >>> jnp.max(x, axis=1)
    Array([9, 7, 8], dtype=int32)

If `keepdims=True`, `ndim` of the output will be same of that of the input.

    >>> jnp.max(x, axis=1, keepdims=True)
    Array([[9],
           [7],
           [8]], dtype=int32)

To include only specific elements in computing the maximum, you can use `where`. It can either have same dimension as input

    >>> where=jnp.array([[0, 0, 1, 0],
    ...                  [0, 0, 1, 1],
    ...                  [1, 1, 1, 0]], dtype=bool)
    >>> jnp.max(x, axis=1, keepdims=True, initial=0, where=where)
    Array([[4],
           [7],
           [8]], dtype=int32)

or must be broadcast compatible with input.

    >>> where = jnp.array([[False],
    ...                    [False],
    ...                    [False]])
    >>> jnp.max(x, axis=0, keepdims=True, initial=0, where=where)
    Array([[0, 0, 0, 0]], dtype=int32)

[](jax.numpy.matvec.html "previous page")

previous

jax.numpy.matvec

[](jax.numpy.maximum.html "next page")

next

jax.numpy.maximum

Contents

- [`max()`](#jax.numpy.max)

By The JAX authors

© Copyright 2024, The JAX Authors.\
