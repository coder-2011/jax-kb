- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.csr_matmat

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.csr_matmat.rst "Download source file")
-  .pdf

# jax.experimental.sparse.csr_matmat

## Contents

- [`csr_matmat()`](#jax.experimental.sparse.csr_matmat)

# jax.experimental.sparse.csr_matmat[\#](#jax-experimental-sparse-csr-matmat "Link to this heading")

jax.experimental.sparse.csr_matmat(*mat*, *B*, *\**, *transpose=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/csr.py#L537-L553)[\#](#jax.experimental.sparse.csr_matmat "Link to this definition")  
Product of CSR sparse matrix and a dense matrix.

Parameters:  
- **mat** ([*CSR*](jax.experimental.sparse.CSR.html#jax.experimental.sparse.CSR "jax.experimental.sparse.CSR")) – CSR matrix

- **B** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – array of shape `(mat.shape[0]`` ``if`` ``transpose`` ``else`` ``mat.shape[1],`` ``cols)` and dtype `mat.dtype`

- **transpose** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – boolean specifying whether to transpose the sparse matrix before computing.

Returns:  
array of shape `(mat.shape[1]`` ``if`` ``transpose`` ``else`` ``mat.shape[0],`` ``cols)`  
representing the matrix vector product.

Return type:  
C

[](jax.experimental.sparse.csr_fromdense.html "previous page")

previous

jax.experimental.sparse.csr_fromdense

[](jax.experimental.sparse.csr_matvec.html "next page")

next

jax.experimental.sparse.csr_matvec

Contents

- [`csr_matmat()`](#jax.experimental.sparse.csr_matmat)

By The JAX authors

© Copyright 2024, The JAX Authors.\
