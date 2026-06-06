- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.csr_fromdense

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.csr_fromdense.rst "Download source file")
-  .pdf

# jax.experimental.sparse.csr_fromdense

## Contents

- [`csr_fromdense()`](#jax.experimental.sparse.csr_fromdense)

# jax.experimental.sparse.csr_fromdense[\#](#jax-experimental-sparse-csr-fromdense "Link to this heading")

jax.experimental.sparse.csr_fromdense(*mat*, *\**, *nse=None*, *index_dtype=\<class 'numpy.int32'\>*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/csr.py#L315-L331)[\#](#jax.experimental.sparse.csr_fromdense "Link to this definition")  
Create a CSR-format sparse matrix from a dense matrix.

Parameters:  
- **mat** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – array to be converted to CSR.

- **nse** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – number of specified entries in `mat`. If not specified, it will be computed from the input matrix.

- **index_dtype** (*DTypeLike*) – dtype of sparse indices

Returns:  
CSR representation of the matrix.

Return type:  
mat_coo

[](jax.experimental.sparse.coo_todense.html "previous page")

previous

jax.experimental.sparse.coo_todense

[](jax.experimental.sparse.csr_matmat.html "next page")

next

jax.experimental.sparse.csr_matmat

Contents

- [`csr_fromdense()`](#jax.experimental.sparse.csr_fromdense)

By The JAX authors

© Copyright 2024, The JAX Authors.\
