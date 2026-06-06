- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.copysign

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.copysign.rst "Download source file")
-  .pdf

# jax.numpy.copysign

## Contents

- [`copysign()`](#jax.numpy.copysign)

# jax.numpy.copysign[\#](#jax-numpy-copysign "Link to this heading")

jax.numpy.copysign(*x1*, *x2*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L2401-L2439)[\#](#jax.numpy.copysign "Link to this definition")  
Copies the sign of each element in `x2` to the corresponding element in `x1`.

JAX implementation of [`numpy.copysign`](https://numpy.org/doc/stable/reference/generated/numpy.copysign.html#numpy.copysign "(in NumPy v2.4)").

Parameters:  
- **x1** (*ArrayLike*) – Input array

- **x2** (*ArrayLike*) – The array whose elements will be used to determine the sign, must be broadcast-compatible with `x1`

Returns:  
An array object containing the potentially changed elements of `x1`, always promotes to inexact dtype, and has a shape of `jnp.broadcast_shapes(x1.shape,`` ``x2.shape)`

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> x1 = jnp.array([5, 2, 0])
    >>> x2 = -1
    >>> jnp.copysign(x1, x2)
    Array([-5., -2., -0.], dtype=float32)

    >>> x1 = jnp.array([6, 8, 0])
    >>> x2 = 2
    >>> jnp.copysign(x1, x2)
    Array([6., 8., 0.], dtype=float32)

    >>> x1 = jnp.array([2, -3])
    >>> x2 = jnp.array([[1],[-4], [5]])
    >>> jnp.copysign(x1, x2)
    Array([[ 2.,  3.],
           [-2., -3.],
           [ 2.,  3.]], dtype=float32)

[](jax.numpy.copy.html "previous page")

previous

jax.numpy.copy

[](jax.numpy.corrcoef.html "next page")

next

jax.numpy.corrcoef

Contents

- [`copysign()`](#jax.numpy.copysign)

By The JAX authors

© Copyright 2024, The JAX Authors.\
