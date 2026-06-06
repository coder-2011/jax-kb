- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fmax

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fmax.rst "Download source file")
-  .pdf

# jax.numpy.fmax

## Contents

- [`fmax()`](#jax.numpy.fmax)

# jax.numpy.fmax[\#](#jax-numpy-fmax "Link to this heading")

jax.numpy.fmax(*x1*, *x2*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L274-L324)[\#](#jax.numpy.fmax "Link to this definition")  
Return element-wise maximum of the input arrays.

JAX implementation of `numpy.fmax()`.

Parameters:  
- **x1** (*ArrayLike*) – input array or scalar

- **x2** (*ArrayLike*) – input array or scalar. x1 and x1 must either have same shape or be broadcast compatible.

Returns:  
An array containing the element-wise maximum of x1 and x2.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

For each pair of elements, `jnp.fmax` returns:  
- the larger of the two if both elements are finite numbers.

- finite number if one element is `nan`.

- `nan` if both elements are `nan`.

- `inf` if one element is `inf` and the other is finite or `nan`.

- `-inf` if one element is `-inf` and the other is `nan`.

Examples

    >>> jnp.fmax(3, 7)
    Array(7, dtype=int32, weak_type=True)
    >>> jnp.fmax(5, jnp.array([1, 7, 9, 4]))
    Array([5, 7, 9, 5], dtype=int32)

    >>> x1 = jnp.array([1, 3, 7, 8])
    >>> x2 = jnp.array([-1, 4, 6, 9])
    >>> jnp.fmax(x1, x2)
    Array([1, 4, 7, 9], dtype=int32)

    >>> x3 = jnp.array([[2, 3, 5, 10],
    ...                 [11, 9, 7, 5]])
    >>> jnp.fmax(x1, x3)
    Array([[ 2,  3,  7, 10],
           [11,  9,  7,  8]], dtype=int32)

    >>> x4 = jnp.array([jnp.inf, 6, -jnp.inf, nan])
    >>> x5 = jnp.array([[3, 5, 7, nan],
    ...                 [nan, 9, nan, -1]])
    >>> jnp.fmax(x4, x5)
    Array([[ inf,   6.,   7.,  nan],
           [ inf,   9., -inf,  -1.]], dtype=float32)

[](jax.numpy.floor_divide.html "previous page")

previous

jax.numpy.floor_divide

[](jax.numpy.fmin.html "next page")

next

jax.numpy.fmin

Contents

- [`fmax()`](#jax.numpy.fmax)

By The JAX authors

© Copyright 2024, The JAX Authors.\
