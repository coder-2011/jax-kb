- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.polar

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.polar.rst "Download source file")
-  .pdf

# jax.scipy.linalg.polar

## Contents

- [`polar()`](#jax.scipy.linalg.polar)

# jax.scipy.linalg.polar[\#](#jax-scipy-linalg-polar "Link to this heading")

jax.scipy.linalg.polar(*a*, *side='right'*, *\**, *method='qdwh'*, *eps=None*, *max_iterations=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L2046-L2143)[\#](#jax.scipy.linalg.polar "Link to this definition")  
Computes the polar decomposition.

Given the \\m \times n\\ matrix \\a\\, returns the factors of the polar decomposition \\u\\ (also \\m \times n\\) and \\p\\ such that \\a = up\\ (if side is `"right"`; \\p\\ is \\n \times n\\) or \\a = pu\\ (if side is `"left"`; \\p\\ is \\m \times m\\), where \\p\\ is positive semidefinite. If \\a\\ is nonsingular, \\p\\ is positive definite and the decomposition is unique. \\u\\ has orthonormal columns unless \\n \> m\\, in which case it has orthonormal rows.

Writing the SVD of \\a\\ as \\a = u\_\mathit{svd} \cdot s\_\mathit{svd} \cdot v^h\_\mathit{svd}\\, we have \\u = u\_\mathit{svd} \cdot v^h\_\mathit{svd}\\. Thus the unitary factor \\u\\ can be constructed as the application of the sign function to the singular values of \\a\\; or, if \\a\\ is Hermitian, the eigenvalues.

Several methods exist to compute the polar decomposition. Currently two are supported:

- `method="svd"`:

  Computes the SVD of \\a\\ and then forms \\u = u\_\mathit{svd} \cdot v^h\_\mathit{svd}\\.

- `method="qdwh"`:

  Applies the [QDWH](https://epubs.siam.org/doi/abs/10.1137/090774999) (QR-based Dynamically Weighted Halley) algorithm.

Parameters:  
- **a** (*ArrayLike*) – array of shape `(...,`` ``m,`` ``n)`.

- **side** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – Determines whether a right or left polar decomposition is computed. If `side` is `"right"` then \\a = up\\. If `side` is `"left"` then \\a = pu\\. The default is `"right"`.

- **method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – Determines the algorithm used, as described above.

- **precision** – [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") object specifying the matmul precision.

- **eps** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* *None*) – The final result will satisfy \\\left\|x_k - x\_{k-1}\right\| \< \left\|x_k\right\| (4\epsilon)^{\frac{1}{3}}\\, where \\x_k\\ are the QDWH iterates. Ignored if `method` is not `"qdwh"`.

- **max_iterations** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – Iterations will terminate after this many steps even if the above is unsatisfied. Ignored if `method` is not `"qdwh"`.

Returns:  
A `(unitary,`` ``posdef)` tuple, where `unitary` is the unitary factor of shape `(...,`` ``m,`` ``n)`, and `posdef` is the positive-semidefinite factor. `posdef` has shape `(...,`` ``n,`` ``n)` or `(...,`` ``m,`` ``m)` depending on whether `side` is `"right"` or `"left"`, respectively.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

Examples

Polar decomposition of a 3x3 matrix:

    >>> a = jnp.array([[1., 2., 3.],
    ...                [5., 4., 2.],
    ...                [3., 2., 1.]])
    >>> U, P = jax.scipy.linalg.polar(a)

U is a Unitary Matrix:

    >>> jnp.round(U.T @ U)  
    Array([[ 1., -0., -0.],
           [-0.,  1.,  0.],
           [-0.,  0.,  1.]], dtype=float32)

P is positive-semidefinite Matrix:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...     print(P)
    [[4.79 3.25 1.23]
     [3.25 3.06 2.01]
     [1.23 2.01 2.91]]

The original matrix can be reconstructed by multiplying the U and P:

    >>> a_reconstructed = U @ P
    >>> jnp.allclose(a, a_reconstructed)
    Array(True, dtype=bool)

[](jax.scipy.linalg.pascal.html "previous page")

previous

jax.scipy.linalg.pascal

[](jax.scipy.linalg.qr.html "next page")

next

jax.scipy.linalg.qr

Contents

- [`polar()`](#jax.scipy.linalg.polar)

By The JAX authors

© Copyright 2024, The JAX Authors.\
