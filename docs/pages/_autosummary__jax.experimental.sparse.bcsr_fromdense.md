- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcsr_fromdense

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcsr_fromdense.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcsr_fromdense

## Contents

- [`bcsr_fromdense()`](#jax.experimental.sparse.bcsr_fromdense)

# jax.experimental.sparse.bcsr_fromdense[\#](#jax-experimental-sparse-bcsr-fromdense "Link to this heading")

jax.experimental.sparse.bcsr_fromdense(*mat*, *\**, *nse=None*, *n_batch=0*, *n_dense=0*, *index_dtype=\<class 'jax.numpy.int32'\>*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcsr.py#L181-L203)[\#](#jax.experimental.sparse.bcsr_fromdense "Link to this definition")  
Create BCSR-format sparse matrix from a dense matrix.

Parameters:  
- **mat** (*ArrayLike*) – array to be converted to BCOO.

- **nse** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – number of stored elements in each batch

- **n_batch** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – number of batch dimensions (default: 0)

- **n_dense** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – number of dense dimensions (default: 0)

- **index_dtype** (*DTypeLike*) – dtype of sparse indices (default: int32)

Returns:  
BCSR representation of the matrix.

Return type:  
mat_bcsr

[](jax.experimental.sparse.bcsr_extract.html "previous page")

previous

jax.experimental.sparse.bcsr_extract

[](jax.experimental.sparse.bcsr_todense.html "next page")

next

jax.experimental.sparse.bcsr_todense

Contents

- [`bcsr_fromdense()`](#jax.experimental.sparse.bcsr_fromdense)

By The JAX authors

© Copyright 2024, The JAX Authors.\
