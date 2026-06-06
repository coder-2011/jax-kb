- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fmin

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fmin.rst "Download source file")
-  .pdf

# jax.numpy.fmin

## Contents

- [`fmin()`](#jax.numpy.fmin)

# jax.numpy.fmin[\#](#jax-numpy-fmin "Link to this heading")

jax.numpy.fmin(*x1*, *x2*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L220-L272)[\#](#jax.numpy.fmin "Link to this definition")  
Return element-wise minimum of the input arrays.

JAX implementation of `numpy.fmin()`.

Parameters:  
- **x1** (*ArrayLike*) – input array or scalar.

- **x2** (*ArrayLike*) – input array or scalar. x1 and x2 must either have same shape or be broadcast compatible.

Returns:  
An array containing the element-wise minimum of x1 and x2.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

For each pair of elements, `jnp.fmin` returns:  
- the smaller of the two if both elements are finite numbers.

- finite number if one element is `nan`.

- `-inf` if one element is `-inf` and the other is finite or `nan`.

- `inf` if one element is `inf` and the other is `nan`.

- `nan` if both elements are `nan`.

Examples

    >>> jnp.fmin(2, 3)
    Array(2, dtype=int32, weak_type=True)
    >>> jnp.fmin(2, jnp.array([1, 4, 2, -1]))
    Array([ 1,  2,  2, -1], dtype=int32)

    >>> x1 = jnp.array([1, 3, 2])
    >>> x2 = jnp.array([2, 1, 4])
    >>> jnp.fmin(x1, x2)
    Array([1, 1, 2], dtype=int32)

    >>> x3 = jnp.array([1, 5, 3])
    >>> x4 = jnp.array([[2, 3, 1],
    ...                 [5, 6, 7]])
    >>> jnp.fmin(x3, x4)
    Array([[1, 3, 1],
           [1, 5, 3]], dtype=int32)

    >>> nan = jnp.nan
    >>> x5 = jnp.array([jnp.inf, 5, nan])
    >>> x6 = jnp.array([[2, 3, nan],
    ...                 [nan, 6, 7]])
    >>> jnp.fmin(x5, x6)
    Array([[ 2.,  3., nan],
           [inf,  5.,  7.]], dtype=float32)

[](jax.numpy.fmax.html "previous page")

previous

jax.numpy.fmax

[](jax.numpy.fmod.html "next page")

next

jax.numpy.fmod

Contents

- [`fmin()`](#jax.numpy.fmin)

By The JAX authors

© Copyright 2024, The JAX Authors.\
