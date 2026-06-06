- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcoo_fromdense

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcoo_fromdense.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcoo_fromdense

## Contents

- [`bcoo_fromdense()`](#jax.experimental.sparse.bcoo_fromdense)

# jax.experimental.sparse.bcoo_fromdense[\#](#jax-experimental-sparse-bcoo-fromdense "Link to this heading")

jax.experimental.sparse.bcoo_fromdense(*mat*, *\**, *nse=None*, *n_batch=0*, *n_dense=0*, *index_dtype=\<class 'jax.numpy.int32'\>*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L248-L270)[\#](#jax.experimental.sparse.bcoo_fromdense "Link to this definition")  
Create BCOO-format sparse matrix from a dense matrix.

Parameters:  
- **mat** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – array to be converted to BCOO.

- **nse** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – number of specified elements in each batch

- **n_batch** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – number of batch dimensions (default: 0)

- **n_dense** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – number of block_dimensions (default: 0)

- **index_dtype** (*DTypeLike*) – dtype of sparse indices (default: int32)

Returns:  
BCOO representation of the matrix.

Return type:  
mat_bcoo

[](jax.experimental.sparse.bcoo_extract.html "previous page")

previous

jax.experimental.sparse.bcoo_extract

[](jax.experimental.sparse.bcoo_gather.html "next page")

next

jax.experimental.sparse.bcoo_gather

Contents

- [`bcoo_fromdense()`](#jax.experimental.sparse.bcoo_fromdense)

By The JAX authors

© Copyright 2024, The JAX Authors.\
