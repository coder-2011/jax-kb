- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.inv

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.inv.rst "Download source file")
-  .pdf

# jax.scipy.linalg.inv

## Contents

- [`inv()`](#jax.scipy.linalg.inv)

# jax.scipy.linalg.inv[\#](#jax-scipy-linalg-inv "Link to this heading")

jax.scipy.linalg.inv(*a*, *overwrite_a=False*, *check_finite=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L580-L634)[\#](#jax.scipy.linalg.inv "Link to this definition")  
Return the inverse of a square matrix

JAX implementation of [`scipy.linalg.inv()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.inv.html#scipy.linalg.inv "(in SciPy v1.19.0.dev)").

Parameters:  
- **a** (*ArrayLike*) – array of shape `(...,`` ``N,`` ``N)` specifying square array(s) to be inverted.

- **overwrite_a** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused in JAX

- **check_finite** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused in JAX

Returns:  
Array of shape `(...,`` ``N,`` ``N)` containing the inverse of the input.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

In most cases, explicitly computing the inverse of a matrix is ill-advised. For example, to compute `x`` ``=`` ``inv(A)`` ``@`` ``b`, it is more performant and numerically precise to use a direct solve, such as [`jax.scipy.linalg.solve()`](jax.scipy.linalg.solve.html#jax.scipy.linalg.solve "jax.scipy.linalg.solve").

See also

- [`jax.numpy.linalg.inv()`](jax.numpy.linalg.inv.html#jax.numpy.linalg.inv "jax.numpy.linalg.inv"): NumPy-style API for matrix inverse

- [`jax.scipy.linalg.solve()`](jax.scipy.linalg.solve.html#jax.scipy.linalg.solve "jax.scipy.linalg.solve"): direct linear solver

Examples

Compute the inverse of a 3x3 matrix

    >>> a = jnp.array([[1., 2., 3.],
    ...                [2., 4., 2.],
    ...                [3., 2., 1.]])
    >>> a_inv = jax.scipy.linalg.inv(a)
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

Note, however, that explicitly computing the inverse in such a case can lead to poor performance and loss of precision as the size of the problem grows. Instead, you should use a direct solver like [`jax.scipy.linalg.solve()`](jax.scipy.linalg.solve.html#jax.scipy.linalg.solve "jax.scipy.linalg.solve"):

    >>> jax.scipy.linalg.solve(a, b)
     Array([ 0.  ,  1.25, -0.5 ], dtype=float32)

[](jax.scipy.linalg.hilbert.html "previous page")

previous

jax.scipy.linalg.hilbert

[](jax.scipy.linalg.invhilbert.html "next page")

next

jax.scipy.linalg.invhilbert

Contents

- [`inv()`](#jax.scipy.linalg.inv)

By The JAX authors

© Copyright 2024, The JAX Authors.\
