- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.ediff1d

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.ediff1d.rst "Download source file")
-  .pdf

# jax.numpy.ediff1d

## Contents

- [`ediff1d()`](#jax.numpy.ediff1d)

# jax.numpy.ediff1d[\#](#jax-numpy-ediff1d "Link to this heading")

jax.numpy.ediff1d(*ary*, *to_end=None*, *to_begin=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L1672-L1733)[\#](#jax.numpy.ediff1d "Link to this definition")  
Compute the differences of the elements of the flattened array.

JAX implementation of [`numpy.ediff1d()`](https://numpy.org/doc/stable/reference/generated/numpy.ediff1d.html#numpy.ediff1d "(in NumPy v2.4)").

Parameters:  
- **ary** (*ArrayLike*) – input array or scalar.

- **to_end** (*ArrayLike* *\|* *None*) – scalar or array, optional, default=None. Specifies the numbers to append to the resulting array.

- **to_begin** (*ArrayLike* *\|* *None*) – scalar or array, optional, default=None. Specifies the numbers to prepend to the resulting array.

Returns:  
An array containing the differences between the elements of the input array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

Unlike NumPy’s implementation of ediff1d, [`jax.numpy.ediff1d()`](#jax.numpy.ediff1d "jax.numpy.ediff1d") will not issue an error if casting `to_end` or `to_begin` to the type of `ary` loses precision.

See also

- [`jax.numpy.diff()`](jax.numpy.diff.html#jax.numpy.diff "jax.numpy.diff"): Computes the n-th order difference between elements of the array along a given axis.

- [`jax.numpy.cumsum()`](jax.numpy.cumsum.html#jax.numpy.cumsum "jax.numpy.cumsum"): Computes the cumulative sum of the elements of the array along a given axis.

- [`jax.numpy.gradient()`](jax.numpy.gradient.html#jax.numpy.gradient "jax.numpy.gradient"): Computes the gradient of an N-dimensional array.

Examples

    >>> a = jnp.array([2, 3, 5, 9, 1, 4])
    >>> jnp.ediff1d(a)
    Array([ 1,  2,  4, -8,  3], dtype=int32)
    >>> jnp.ediff1d(a, to_begin=-10)
    Array([-10,   1,   2,   4,  -8,   3], dtype=int32)
    >>> jnp.ediff1d(a, to_end=jnp.array([20, 30]))
    Array([ 1,  2,  4, -8,  3, 20, 30], dtype=int32)
    >>> jnp.ediff1d(a, to_begin=-10, to_end=jnp.array([20, 30]))
    Array([-10,   1,   2,   4,  -8,   3,  20,  30], dtype=int32)

For array with `ndim`` ``>`` ``1`, the differences are computed after flattening the input array.

    >>> a1 = jnp.array([[2, -1, 4, 7],
    ...                 [3, 5, -6, 9]])
    >>> jnp.ediff1d(a1)
    Array([ -3,   5,   3,  -4,   2, -11,  15], dtype=int32)
    >>> a2 = jnp.array([2, -1, 4, 7, 3, 5, -6, 9])
    >>> jnp.ediff1d(a2)
    Array([ -3,   5,   3,  -4,   2, -11,  15], dtype=int32)

[](jax.numpy.dtype.html "previous page")

previous

jax.numpy.dtype

[](jax.numpy.einsum.html "next page")

next

jax.numpy.einsum

Contents

- [`ediff1d()`](#jax.numpy.ediff1d)

By The JAX authors

© Copyright 2024, The JAX Authors.\
