- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.coo_matvec

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.coo_matvec.rst "Download source file")
-  .pdf

# jax.experimental.sparse.coo_matvec

## Contents

- [`coo_matvec()`](#jax.experimental.sparse.coo_matvec)

# jax.experimental.sparse.coo_matvec[\#](#jax-experimental-sparse-coo-matvec "Link to this heading")

jax.experimental.sparse.coo_matvec(*mat*, *v*, *transpose=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/coo.py#L397-L413)[\#](#jax.experimental.sparse.coo_matvec "Link to this definition")  
Product of COO sparse matrix and a dense vector.

Parameters:  
- **mat** ([*COO*](jax.experimental.sparse.COO.html#jax.experimental.sparse.COO "jax.experimental.sparse.COO")) – COO matrix

- **v** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – one-dimensional array of size `(shape[0]`` ``if`` ``transpose`` ``else`` ``shape[1],)` and dtype `mat.dtype`

- **transpose** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – boolean specifying whether to transpose the sparse matrix before computing.

Returns:  
array of shape `(mat.shape[1]`` ``if`` ``transpose`` ``else`` ``mat.shape[0],)` representing  
the matrix vector product.

Return type:  
y

[](jax.experimental.sparse.coo_matmat.html "previous page")

previous

jax.experimental.sparse.coo_matmat

[](jax.experimental.sparse.coo_todense.html "next page")

next

jax.experimental.sparse.coo_todense

Contents

- [`coo_matvec()`](#jax.experimental.sparse.coo_matvec)

By The JAX authors

© Copyright 2024, The JAX Authors.\
