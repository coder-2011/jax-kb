- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.linalg.spsolve

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.linalg.spsolve.rst "Download source file")
-  .pdf

# jax.experimental.sparse.linalg.spsolve

## Contents

- [`spsolve()`](#jax.experimental.sparse.linalg.spsolve)

# jax.experimental.sparse.linalg.spsolve[\#](#jax-experimental-sparse-linalg-spsolve "Link to this heading")

jax.experimental.sparse.linalg.spsolve(*data*, *indices*, *indptr*, *b*, *tol=1e-06*, *reorder=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/linalg.py#L603-L626)[\#](#jax.experimental.sparse.linalg.spsolve "Link to this definition")  
A sparse direct solver using QR factorization.

Accepts a sparse matrix in CSR format data, indices, indptr arrays. Currently only the CUDA GPU backend is implemented, the CPU backend will fall back to scipy.sparse.linalg.spsolve. Neither the CPU nor the GPU implementation support batching with vmap.

Parameters:  
- **data** – An array containing the non-zero entries of the CSR matrix.

- **indices** – The column indices of the CSR matrix.

- **indptr** – The row pointer array of the CSR matrix.

- **b** – The right hand side of the linear system.

- **tol** – Tolerance to decide if singular or not. Defaults to 1e-6.

- **reorder** – The reordering scheme to use to reduce fill-in. No reordering if `reorder=0`. Otherwise, symrcm, symamd, or csrmetisnd (`reorder=1,2,3`), respectively. Defaults to symrcm.

Returns:  
An array with the same dtype and size as b representing the solution to the sparse linear system.

[](jax.experimental.sparse.csr_todense.html "previous page")

previous

jax.experimental.sparse.csr_todense

[](jax.experimental.sparse.linalg.lobpcg_standard.html "next page")

next

jax.experimental.sparse.linalg.lobpcg_standard

Contents

- [`spsolve()`](#jax.experimental.sparse.linalg.spsolve)

By The JAX authors

© Copyright 2024, The JAX Authors.\
