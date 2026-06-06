- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.min

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.min.rst "Download source file")
-  .pdf

# jax.numpy.min

## Contents

- [`min()`](#jax.numpy.min)

# jax.numpy.min[\#](#jax-numpy-min "Link to this heading")

jax.numpy.min(*a*, *axis=None*, *out=None*, *keepdims=False*, *initial=None*, *where=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L491-L563)[\#](#jax.numpy.min "Link to this definition")  
Return the minimum of array elements along a given axis.

JAX implementation of [`numpy.min()`](https://numpy.org/doc/stable/reference/generated/numpy.min.html#numpy.min "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – Input array.

- **axis** (*Axis*) – int or array, default=None. Axis along which the minimum to be computed. If None, the minimum is computed along all the axes.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If true, reduced axes are left in the result with size 1.

- **initial** (*ArrayLike* *\|* *None*) – int or array, Default=None. Initial value for the minimum.

- **where** (*ArrayLike* *\|* *None*) – int or array, default=None. The elements to be used in the minimum. Array should be broadcast compatible to the input. `initial` must be specified when `where` is used.

- **out** (*None*) – Unused by JAX.

Returns:  
An array of minimum values along the given axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.max()`](jax.numpy.max.html#jax.numpy.max "jax.numpy.max"): Compute the maximum of array elements along a given axis.

- [`jax.numpy.sum()`](jax.numpy.sum.html#jax.numpy.sum "jax.numpy.sum"): Compute the sum of array elements along a given axis.

- [`jax.numpy.prod()`](jax.numpy.prod.html#jax.numpy.prod "jax.numpy.prod"): Compute the product of array elements along a given axis.

Examples

By default, the minimum is computed along all the axes.

    >>> x = jnp.array([[2, 5, 1, 6],
    ...                [3, -7, -2, 4],
    ...                [8, -4, 1, -3]])
    >>> jnp.min(x)
    Array(-7, dtype=int32)

If `axis=1`, the minimum is computed along axis 1.

    >>> jnp.min(x, axis=1)
    Array([ 1, -7, -4], dtype=int32)

If `keepdims=True`, `ndim` of the output will be same of that of the input.

    >>> jnp.min(x, axis=1, keepdims=True)
    Array([[ 1],
           [-7],
           [-4]], dtype=int32)

To include only specific elements in computing the minimum, you can use `where`. `where` can either have same dimension as input.

    >>> where=jnp.array([[1, 0, 1, 0],
    ...                  [0, 0, 1, 1],
    ...                  [1, 1, 1, 0]], dtype=bool)
    >>> jnp.min(x, axis=1, keepdims=True, initial=0, where=where)
    Array([[ 0],
           [-2],
           [-4]], dtype=int32)

or must be broadcast compatible with input.

    >>> where = jnp.array([[False],
    ...                    [False],
    ...                    [False]])
    >>> jnp.min(x, axis=0, keepdims=True, initial=0, where=where)
    Array([[0, 0, 0, 0]], dtype=int32)

[](jax.numpy.mgrid.html "previous page")

previous

jax.numpy.mgrid

[](jax.numpy.minimum.html "next page")

next

jax.numpy.minimum

Contents

- [`min()`](#jax.numpy.min)

By The JAX authors

© Copyright 2024, The JAX Authors.\
