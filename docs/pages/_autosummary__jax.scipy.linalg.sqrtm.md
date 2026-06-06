- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.sqrtm

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.sqrtm.rst "Download source file")
-  .pdf

# jax.scipy.linalg.sqrtm

## Contents

- [`sqrtm()`](#jax.scipy.linalg.sqrtm)

# jax.scipy.linalg.sqrtm[\#](#jax-scipy-linalg-sqrtm "Link to this heading")

jax.scipy.linalg.sqrtm(*A*, *blocksize=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L2179-L2226)[\#](#jax.scipy.linalg.sqrtm "Link to this definition")  
Compute the matrix square root

This function is implemented using [`scipy.linalg.schur()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.schur.html#scipy.linalg.schur "(in SciPy v1.19.0.dev)"), which is only supported on CPU.

JAX implementation of [`scipy.linalg.sqrtm()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.sqrtm.html#scipy.linalg.sqrtm "(in SciPy v1.19.0.dev)").

Parameters:  
- **A** (*ArrayLike*) – array of shape `(N,`` ``N)`

- **blocksize** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Not supported in JAX; JAX always uses `blocksize=1`.

Returns:  
An array of shape `(N,`` ``N)` containing the matrix square root of `A`

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.linalg.expm()`](jax.scipy.linalg.expm.html#jax.scipy.linalg.expm "jax.scipy.linalg.expm")

Examples

    >>> a = jnp.array([[1., 2., 3.],
    ...                [2., 4., 2.],
    ...                [3., 2., 1.]])
    >>> sqrt_a = jax.scipy.linalg.sqrtm(a)
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(sqrt_a)
    [[0.92+0.71j 0.54+0.j   0.92-0.71j]
     [0.54+0.j   1.85+0.j   0.54-0.j  ]
     [0.92-0.71j 0.54-0.j   0.92+0.71j]]

By definition, matrix multiplication of the matrix square root with itself should equal the input:

    >>> jnp.allclose(a, sqrt_a @ sqrt_a)
    Array(True, dtype=bool)

Notes

This function implements the complex Schur method described in [^1]. It does not use recursive blocking to speed up computations as a Sylvester Equation solver is not yet available in JAX.

References

\[[1](#id1)\]

Björck, Å., & Hammarling, S. (1983). “A Schur method for the square root of a matrix”. Linear algebra and its applications, 52, 127-140.

[](jax.scipy.linalg.solve_triangular.html "previous page")

previous

jax.scipy.linalg.solve_triangular

[](jax.scipy.linalg.svd.html "next page")

next

jax.scipy.linalg.svd

Contents

- [`sqrtm()`](#jax.scipy.linalg.sqrtm)

By The JAX authors

© Copyright 2024, The JAX Authors.\

[^1]:
