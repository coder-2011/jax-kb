- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcoo_sum_duplicates

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcoo_sum_duplicates.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcoo_sum_duplicates

## Contents

- [`bcoo_sum_duplicates()`](#jax.experimental.sparse.bcoo_sum_duplicates)

# jax.experimental.sparse.bcoo_sum_duplicates[\#](#jax-experimental-sparse-bcoo-sum-duplicates "Link to this heading")

jax.experimental.sparse.bcoo_sum_duplicates(*mat*, *nse=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L1391-L1409)[\#](#jax.experimental.sparse.bcoo_sum_duplicates "Link to this definition")  
Sums duplicate indices within a BCOO array, returning an array with sorted indices.

Parameters:  
- **mat** ([*BCOO*](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")) – BCOO array

- **nse** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – integer (optional). The number of specified elements in the output matrix. This must be specified for bcoo_sum_duplicates to be compatible with JIT and other JAX transformations. If not specified, the optimal nse will be computed based on the contents of the data and index arrays. If specified nse is larger than necessary, data and index arrays will be padded with standard fill values. If smaller than necessary, data elements will be dropped from the output matrix.

Returns:  
BCOO array with sorted indices and no duplicate indices.

Return type:  
mat_out

[](jax.experimental.sparse.bcoo_squeeze.html "previous page")

previous

jax.experimental.sparse.bcoo_squeeze

[](jax.experimental.sparse.bcoo_todense.html "next page")

next

jax.experimental.sparse.bcoo_todense

Contents

- [`bcoo_sum_duplicates()`](#jax.experimental.sparse.bcoo_sum_duplicates)

By The JAX authors

© Copyright 2024, The JAX Authors.\
