- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcoo_squeeze

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcoo_squeeze.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcoo_squeeze

## Contents

- [`bcoo_squeeze()`](#jax.experimental.sparse.bcoo_squeeze)

# jax.experimental.sparse.bcoo_squeeze[\#](#jax-experimental-sparse-bcoo-squeeze "Link to this heading")

jax.experimental.sparse.bcoo_squeeze(*arr*, *\**, *dimensions*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L1931-L1957)[\#](#jax.experimental.sparse.bcoo_squeeze "Link to this definition")  
Sparse implementation of [`jax.lax.squeeze()`](jax.lax.squeeze.html#jax.lax.squeeze "jax.lax.squeeze").

Squeeze any number of size 1 dimensions from an array.

Parameters:  
- **arr** ([*BCOO*](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")) – BCOO array to be reshaped.

- **dimensions** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – sequence of integers specifying dimensions to squeeze.

Returns:  
reshaped array.

Return type:  
out

[](jax.experimental.sparse.bcoo_sort_indices.html "previous page")

previous

jax.experimental.sparse.bcoo_sort_indices

[](jax.experimental.sparse.bcoo_sum_duplicates.html "next page")

next

jax.experimental.sparse.bcoo_sum_duplicates

Contents

- [`bcoo_squeeze()`](#jax.experimental.sparse.bcoo_squeeze)

By The JAX authors

© Copyright 2024, The JAX Authors.\
