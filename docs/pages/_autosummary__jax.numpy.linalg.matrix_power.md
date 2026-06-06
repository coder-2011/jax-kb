- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.matrix_power

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.matrix_power.rst "Download source file")
-  .pdf

# jax.numpy.linalg.matrix_power

## Contents

- [`matrix_power()`](#jax.numpy.linalg.matrix_power)

# jax.numpy.linalg.matrix_power[\#](#jax-numpy-linalg-matrix-power "Link to this heading")

jax.numpy.linalg.matrix_power(*a*, *n*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L332-L412)[\#](#jax.numpy.linalg.matrix_power "Link to this definition")  
Raise a square matrix to an integer power.

JAX implementation of [`numpy.linalg.matrix_power()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_power.html#numpy.linalg.matrix_power "(in NumPy v2.4)"), implemented via repeated squarings.

Parameters:  
- **a** (*ArrayLike*) – array of shape `(...,`` ``M,`` ``M)` to be raised to the power n.

- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the integer exponent to which the matrix should be raised.

Returns:  
Array of shape `(...,`` ``M,`` ``M)` containing the matrix power of a to the n.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> a = jnp.array([[1., 2.],
    ...                [3., 4.]])
    >>> jnp.linalg.matrix_power(a, 3)
    Array([[ 37.,  54.],
           [ 81., 118.]], dtype=float32)
    >>> a @ a @ a  # equivalent evaluated directly
    Array([[ 37.,  54.],
           [ 81., 118.]], dtype=float32)

This also supports zero powers:

    >>> jnp.linalg.matrix_power(a, 0)
    Array([[1., 0.],
           [0., 1.]], dtype=float32)

and also supports negative powers:

    >>> with jnp.printoptions(precision=3):
    ...   jnp.linalg.matrix_power(a, -2)
    Array([[ 5.5 , -2.5 ],
           [-3.75,  1.75]], dtype=float32)

Negative powers are equivalent to matmul of the inverse:

    >>> inv_a = jnp.linalg.inv(a)
    >>> with jnp.printoptions(precision=3):
    ...   inv_a @ inv_a
    Array([[ 5.5 , -2.5 ],
           [-3.75,  1.75]], dtype=float32)

[](jax.numpy.linalg.matrix_norm.html "previous page")

previous

jax.numpy.linalg.matrix_norm

[](jax.numpy.linalg.matrix_rank.html "next page")

next

jax.numpy.linalg.matrix_rank

Contents

- [`matrix_power()`](#jax.numpy.linalg.matrix_power)

By The JAX authors

© Copyright 2024, The JAX Authors.\
