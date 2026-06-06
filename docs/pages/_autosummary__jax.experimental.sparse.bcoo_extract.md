- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcoo_extract

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcoo_extract.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcoo_extract

## Contents

- [`bcoo_extract()`](#jax.experimental.sparse.bcoo_extract)

# jax.experimental.sparse.bcoo_extract[\#](#jax-experimental-sparse-bcoo-extract "Link to this heading")

jax.experimental.sparse.bcoo_extract(*sparr*, *arr*, *\**, *assume_unique=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L371-L394)[\#](#jax.experimental.sparse.bcoo_extract "Link to this definition")  
Extract values from a dense array according to the sparse array’s indices.

Parameters:  
- **sparr** ([*BCOO*](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")) – BCOO array whose indices will be used for the output.

- **arr** (*ArrayLike*) – ArrayLike with shape equal to self.shape

- **assume_unique** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None*) – bool, defaults to sparr.unique_indices If True, extract values for every index, even if index contains duplicates. If False, duplicate indices will have their values summed and returned in the position of the first index.

Returns:  
a BCOO array with the same sparsity pattern as self.

Return type:  
extracted

[](jax.experimental.sparse.bcoo_dynamic_slice.html "previous page")

previous

jax.experimental.sparse.bcoo_dynamic_slice

[](jax.experimental.sparse.bcoo_fromdense.html "next page")

next

jax.experimental.sparse.bcoo_fromdense

Contents

- [`bcoo_extract()`](#jax.experimental.sparse.bcoo_extract)

By The JAX authors

© Copyright 2024, The JAX Authors.\
