- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.diff

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.diff.rst "Download source file")
-  .pdf

# jax.numpy.diff

## Contents

- [`diff()`](#jax.numpy.diff)

# jax.numpy.diff[\#](#jax-numpy-diff "Link to this heading")

jax.numpy.diff(*a*, *n=1*, *axis=-1*, *prepend=None*, *append=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L1560-L1670)[\#](#jax.numpy.diff "Link to this definition")  
Calculate n-th order difference between array elements along a given axis.

JAX implementation of [`numpy.diff()`](https://numpy.org/doc/stable/reference/generated/numpy.diff.html#numpy.diff "(in NumPy v2.4)").

The first order difference is computed by `a[i+1]`` ``-`` ``a[i]`, and the n-th order difference is computed `n` times recursively.

Parameters:  
- **a** (*ArrayLike*) – input array. Must have `a.ndim`` ``>=`` ``1`.

- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int, optional, default=1. Order of the difference. Specifies the number of times the difference is computed. If n=0, no difference is computed and input is returned as is.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int, optional, default=-1. Specifies the axis along which the difference is computed. The difference is computed along `axis`` ``-1` by default.

- **prepend** (*ArrayLike* *\|* *None*) – scalar or array, optional, default=None. Specifies the values to be prepended along `axis` before computing the difference.

- **append** (*ArrayLike* *\|* *None*) – scalar or array, optional, default=None. Specifies the values to be appended along `axis` before computing the difference.

Returns:  
An array containing the n-th order difference between the elements of `a`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.ediff1d()`](jax.numpy.ediff1d.html#jax.numpy.ediff1d "jax.numpy.ediff1d"): Computes the differences between consecutive elements of an array.

- [`jax.numpy.cumsum()`](jax.numpy.cumsum.html#jax.numpy.cumsum "jax.numpy.cumsum"): Computes the cumulative sum of the elements of the array along a given axis.

- [`jax.numpy.gradient()`](jax.numpy.gradient.html#jax.numpy.gradient "jax.numpy.gradient"): Computes the gradient of an N-dimensional array.

Examples

`jnp.diff` computes the first order difference along `axis`, by default.

    >>> a = jnp.array([[1, 5, 2, 9],
    ...                [3, 8, 7, 4]])
    >>> jnp.diff(a)
    Array([[ 4, -3,  7],
           [ 5, -1, -3]], dtype=int32)

When `n`` ``=`` ``2`, second order difference is computed along `axis`.

    >>> jnp.diff(a, n=2)
    Array([[-7, 10],
           [-6, -2]], dtype=int32)

When `prepend`` ``=`` ``2`, it is prepended to `a` along `axis` before computing the difference.

    >>> jnp.diff(a, prepend=2)
    Array([[-1,  4, -3,  7],
           [ 1,  5, -1, -3]], dtype=int32)

When `append`` ``=`` ``jnp.array([[3],[1]])`, it is appended to `a` along `axis` before computing the difference.

    >>> jnp.diff(a, append=jnp.array([[3],[1]]))
    Array([[ 4, -3,  7, -6],
           [ 5, -1, -3, -3]], dtype=int32)

[](jax.numpy.diagonal.html "previous page")

previous

jax.numpy.diagonal

[](jax.numpy.digitize.html "next page")

next

jax.numpy.digitize

Contents

- [`diff()`](#jax.numpy.diff)

By The JAX authors

© Copyright 2024, The JAX Authors.\
