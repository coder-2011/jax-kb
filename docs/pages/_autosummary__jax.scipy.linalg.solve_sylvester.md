- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.solve_sylvester

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.solve_sylvester.rst "Download source file")
-  .pdf

# jax.scipy.linalg.solve_sylvester

## Contents

- [`solve_sylvester()`](#jax.scipy.linalg.solve_sylvester)

# jax.scipy.linalg.solve_sylvester[\#](#jax-scipy-linalg-solve-sylvester "Link to this heading")

jax.scipy.linalg.solve_sylvester(*A*, *B*, *C*, *\**, *method='schur'*, *tol=1e-08*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L3302-L3370)[\#](#jax.scipy.linalg.solve_sylvester "Link to this definition")  
Solves the Sylvester equation .. math:

    AX + XB = C

Using one of two methods.

1.  Bartell-Stewart (schur) algorithm (default) \[CPU ONLY\]:

Where A and B are first decomposed using Schur decomposition to construct and alternate sylvester equation: .. math:

    RY + YS^T = F

Where R and S are in quasitriangular form when A and B are real valued and triangular when A and B are complex.

2.  The Eigen decomposition algorithm \[CPU and GPU\]

Parameters:  
- **A** (*ArrayLike*) – array of shape `(...,`` ``m,`` ``m)`

- **B** (*ArrayLike*) – array of shape `(...,`` ``n,`` ``n)`

- **C** (*ArrayLike*) – array of shape `(...,`` ``m,`` ``n)`. Batch dimensions are broadcast across `A`, `B`, and `C`.

- **method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – “schur” is the default and is accurate but slow, and “eigen” is an alternative that is faster but less accurate for ill-conditioned matrices.

- **tol** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) – How close the sum of the eigenvalues from A and B can be to zero before returning matrix of NaNs

Returns:  
Array of shape `(...,`` ``m,`` ``n)` representing the solution `X`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> A = jax.numpy.array([[1, 2], [3, 4]])
    >>> B = jax.numpy.array([[5, 6], [7, 8]])
    >>> C = jax.numpy.array([[6, 8], [10, 12]])
    >>> X = jax.scipy.linalg.solve_sylvester(A, B, C)
    >>> print(X) 
    [[1. 0.]
     [0.  1.]]

Notes

The Bartel-Stewart algorithm is robust because a Schur decomposition always exists even for defective matrices, and it handles complex and ill-conditioned problems better than the eigen decomposition method. However, there are a couple of drawbacks. First, It is computationally more expensive than the eigen decomposition method because you need to perform a Schur decomposition and then scan the entire solution matrix. Second, it requires more system memory compared to the eigen decomposition method.

The eigen decomposition method is the fastest method to solve a sylvester equation. However, this speed brings with it a couple of drawbacks. First, A and B must be diagonalizable otherwise the eigenvectors will be linearly dependent and ill-conditioned leading to accuracy issues. Second, when the eigenvectors are not orthogonal roundoff errors are amplified.

Additionally, for complex types as the size of the matrix increases the accuracy of the results degrades. Float64 types are most robust to degradation.

The tol argument allows you to specify how ill-conditioned a matrix can be and still estimate a solution. For matrices that are ill-conditioned we recommend using float64 instead of the default float32 dtype. The solver can still return good estimates for ill-conditioned matrices depending on how close to zero the sums of the eigenvalues of A and B are.

[](jax.scipy.linalg.solve.html "previous page")

previous

jax.scipy.linalg.solve

[](jax.scipy.linalg.solve_triangular.html "next page")

next

jax.scipy.linalg.solve_triangular

Contents

- [`solve_sylvester()`](#jax.scipy.linalg.solve_sylvester)

By The JAX authors

© Copyright 2024, The JAX Authors.\
