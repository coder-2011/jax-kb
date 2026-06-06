- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.coo_matmat

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.coo_matmat.rst "Download source file")
-  .pdf

# jax.experimental.sparse.coo_matmat

## Contents

- [`coo_matmat()`](#jax.experimental.sparse.coo_matmat)

# jax.experimental.sparse.coo_matmat[\#](#jax-experimental-sparse-coo-matmat "Link to this heading")

jax.experimental.sparse.coo_matmat(*mat*, *B*, *\**, *transpose=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/coo.py#L520-L536)[\#](#jax.experimental.sparse.coo_matmat "Link to this definition")  
Product of COO sparse matrix and a dense matrix.

Parameters:  
- **mat** ([*COO*](jax.experimental.sparse.COO.html#jax.experimental.sparse.COO "jax.experimental.sparse.COO")) – COO matrix

- **B** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – array of shape `(mat.shape[0]`` ``if`` ``transpose`` ``else`` ``mat.shape[1],`` ``cols)` and dtype `mat.dtype`

- **transpose** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – boolean specifying whether to transpose the sparse matrix before computing.

Returns:  
array of shape `(mat.shape[1]`` ``if`` ``transpose`` ``else`` ``mat.shape[0],`` ``cols)`  
representing the matrix vector product.

Return type:  
C

[](jax.experimental.sparse.coo_fromdense.html "previous page")

previous

jax.experimental.sparse.coo_fromdense

[](jax.experimental.sparse.coo_matvec.html "next page")

next

jax.experimental.sparse.coo_matvec

Contents

- [`coo_matmat()`](#jax.experimental.sparse.coo_matmat)

By The JAX authors

© Copyright 2024, The JAX Authors.\
