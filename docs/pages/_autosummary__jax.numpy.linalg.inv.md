- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.inv

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.inv.rst "Download source file")
-  .pdf

# jax.numpy.linalg.inv

## Contents

- [`inv()`](#jax.numpy.linalg.inv)

# jax.numpy.linalg.inv[\#](#jax-numpy-linalg-inv "Link to this heading")

jax.numpy.linalg.inv(*a*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L1092-L1150)[\#](#jax.numpy.linalg.inv "Link to this definition")  
Return the inverse of a square matrix

JAX implementation of [`numpy.linalg.inv()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html#numpy.linalg.inv "(in NumPy v2.4)").

Parameters:  
**a** (*ArrayLike*) – array of shape `(...,`` ``N,`` ``N)` specifying square array(s) to be inverted.

Returns:  
Array of shape `(...,`` ``N,`` ``N)` containing the inverse of the input.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

In most cases, explicitly computing the inverse of a matrix is ill-advised. For example, to compute `x`` ``=`` ``inv(A)`` ``@`` ``b`, it is more performant and numerically precise to use a direct solve, such as [`jax.scipy.linalg.solve()`](jax.scipy.linalg.solve.html#jax.scipy.linalg.solve "jax.scipy.linalg.solve").

See also

- [`jax.scipy.linalg.inv()`](jax.scipy.linalg.inv.html#jax.scipy.linalg.inv "jax.scipy.linalg.inv"): SciPy-style API for matrix inverse

- [`jax.numpy.linalg.solve()`](jax.numpy.linalg.solve.html#jax.numpy.linalg.solve "jax.numpy.linalg.solve"): direct linear solver

Examples

Compute the inverse of a 3x3 matrix

    >>> a = jnp.array([[1., 2., 3.],
    ...                [2., 4., 2.],
    ...                [3., 2., 1.]])
    >>> a_inv = jnp.linalg.inv(a)
    >>> a_inv  
    Array([[ 0.        , -0.25      ,  0.5       ],
           [-0.25      ,  0.5       , -0.25000003],
           [ 0.5       , -0.25      ,  0.        ]], dtype=float32)

Check that multiplying with the inverse gives the identity:

    >>> jnp.allclose(a @ a_inv, jnp.eye(3), atol=1E-5)
    Array(True, dtype=bool)

Multiply the inverse by a vector `b`, to find a solution to `a`` ``@`` ``x`` ``=`` ``b`

    >>> b = jnp.array([1., 4., 2.])
    >>> a_inv @ b
    Array([ 0.  ,  1.25, -0.5 ], dtype=float32)

Note, however, that explicitly computing the inverse in such a case can lead to poor performance and loss of precision as the size of the problem grows. Instead, you should use a direct solver like [`jax.numpy.linalg.solve()`](jax.numpy.linalg.solve.html#jax.numpy.linalg.solve "jax.numpy.linalg.solve"):

    >>> jnp.linalg.solve(a, b)
     Array([ 0.  ,  1.25, -0.5 ], dtype=float32)

[](jax.numpy.linalg.eigvalsh.html "previous page")

previous

jax.numpy.linalg.eigvalsh

[](jax.numpy.linalg.lstsq.html "next page")

next

jax.numpy.linalg.lstsq

Contents

- [`inv()`](#jax.numpy.linalg.inv)

By The JAX authors

© Copyright 2024, The JAX Authors.\
