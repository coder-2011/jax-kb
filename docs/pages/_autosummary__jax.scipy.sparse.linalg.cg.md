- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.sparse.linalg.cg

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.sparse.linalg.cg.rst "Download source file")
-  .pdf

# jax.scipy.sparse.linalg.cg

## Contents

- [`cg()`](#jax.scipy.sparse.linalg.cg)

# jax.scipy.sparse.linalg.cg[\#](#jax-scipy-sparse-linalg-cg "Link to this heading")

jax.scipy.sparse.linalg.cg(*A*, *b*, *x0=None*, *\**, *tol=1e-05*, *atol=0.0*, *maxiter=None*, *M=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/sparse/linalg.py#L233-L289)[\#](#jax.scipy.sparse.linalg.cg "Link to this definition")  
Use Conjugate Gradient iteration to solve `Ax`` ``=`` ``b`.

The numerics of JAX’s `cg` should exact match SciPy’s `cg` (up to numerical precision), but note that the interface is slightly different: you need to supply the linear operator `A` as a function instead of a sparse matrix or `LinearOperator`.

Derivatives of `cg` are implemented via implicit differentiation with another `cg` solve, rather than by differentiating *through* the solver. They will be accurate only if both solves converge.

Parameters:  
- **A** (*ndarray,* *function, or* *matmul-compatible object*) – 2D array or function that calculates the linear map (matrix-vector product) `Ax` when called like `A(x)` or `A`` ``@`` ``x`. `A` must represent a hermitian, positive definite matrix, and must return array(s) with the same structure and shape as its argument.

- **b** (*array* *or* *tree* *of* *arrays*) – Right hand side of the linear system representing a single vector. Can be stored as an array or Python container of array(s) with any shape.

- **x0** (*array* *or* *tree* *of* *arrays*) – Starting guess for the solution. Must have the same structure as `b`.

- **tol** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")*,* *optional*) – Tolerances for convergence, `norm(residual)`` ``<=`` ``max(tol*norm(b),`` ``atol)`. We do not implement SciPy’s “legacy” behavior, so JAX’s tolerance will differ from SciPy unless you explicitly pass `atol` to SciPy’s `cg`.

- **atol** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")*,* *optional*) – Tolerances for convergence, `norm(residual)`` ``<=`` ``max(tol*norm(b),`` ``atol)`. We do not implement SciPy’s “legacy” behavior, so JAX’s tolerance will differ from SciPy unless you explicitly pass `atol` to SciPy’s `cg`.

- **maxiter** ([*integer*](jax.numpy.integer.html#jax.numpy.integer "jax.numpy.integer")) – Maximum number of iterations. Iteration will stop after maxiter steps even if the specified tolerance has not been achieved.

- **M** (*ndarray,* *function, or* *matmul-compatible object*) – Preconditioner for A. The preconditioner should approximate the inverse of A. Effective preconditioning dramatically improves the rate of convergence, which implies that fewer iterations are needed to reach a given error tolerance.

Returns:  
- **x** (*array or tree of arrays*) – The converged solution. Has the same structure as `b`.

- **info** (*None*) – Placeholder for convergence information. In the future, JAX will report the number of iterations when convergence is not achieved, like SciPy.

See also

[`scipy.sparse.linalg.cg`](https://scipy.github.io/devdocs/reference/generated/scipy.sparse.linalg.cg.html#scipy.sparse.linalg.cg "(in SciPy v1.19.0.dev)"), [`jax.lax.custom_linear_solve`](jax.lax.custom_linear_solve.html#jax.lax.custom_linear_solve "jax.lax.custom_linear_solve")

[](jax.scipy.sparse.linalg.bicgstab.html "previous page")

previous

jax.scipy.sparse.linalg.bicgstab

[](jax.scipy.sparse.linalg.gmres.html "next page")

next

jax.scipy.sparse.linalg.gmres

Contents

- [`cg()`](#jax.scipy.sparse.linalg.cg)

By The JAX authors

© Copyright 2024, The JAX Authors.\
