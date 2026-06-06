- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.coo_fromdense

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.coo_fromdense.rst "Download source file")
-  .pdf

# jax.experimental.sparse.coo_fromdense

## Contents

- [`coo_fromdense()`](#jax.experimental.sparse.coo_fromdense)

# jax.experimental.sparse.coo_fromdense[\#](#jax-experimental-sparse-coo-fromdense "Link to this heading")

jax.experimental.sparse.coo_fromdense(*mat*, *\**, *nse=None*, *index_dtype=\<class 'jax.numpy.int32'\>*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/coo.py#L287-L304)[\#](#jax.experimental.sparse.coo_fromdense "Link to this definition")  
Create a COO-format sparse matrix from a dense matrix.

Parameters:  
- **mat** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – array to be converted to COO.

- **nse** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – number of specified entries in `mat`. If not specified, it will be computed from the input matrix.

- **index_dtype** (*DTypeLike*) – dtype of sparse indices

Returns:  
COO representation of the matrix.

Return type:  
mat_coo

[](jax.experimental.sparse.CSR.html "previous page")

previous

jax.experimental.sparse.CSR

[](jax.experimental.sparse.coo_matmat.html "next page")

next

jax.experimental.sparse.coo_matmat

Contents

- [`coo_fromdense()`](#jax.experimental.sparse.coo_fromdense)

By The JAX authors

© Copyright 2024, The JAX Authors.\
