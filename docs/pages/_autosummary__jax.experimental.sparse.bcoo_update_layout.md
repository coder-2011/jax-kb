- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcoo_update_layout

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcoo_update_layout.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcoo_update_layout

## Contents

- [`bcoo_update_layout()`](#jax.experimental.sparse.bcoo_update_layout)

# jax.experimental.sparse.bcoo_update_layout[\#](#jax-experimental-sparse-bcoo-update-layout "Link to this heading")

jax.experimental.sparse.bcoo_update_layout(*mat*, *\**, *n_batch=None*, *n_dense=None*, *on_inefficient='error'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L1572-L1705)[\#](#jax.experimental.sparse.bcoo_update_layout "Link to this definition")  
Update the storage layout (i.e. n_batch & n_dense) of a BCOO matrix.

In many cases this can be done without introducing undue storage overhead. However, increasing `mat.n_batch` or `mat.n_dense` will lead to very inefficient storage, with many explicitly-stored zeros, unless the new batch or dense dimensions have size 0 or 1. In such cases, `bcoo_update_layout` will raise a `SparseEfficiencyError`. This can be silenced by specifying the `on_inefficient` argument.

Parameters:  
- **mat** ([*BCOO*](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")) – BCOO array

- **n_batch** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional(int) the number of batch dimensions in the output matrix. If None, then n_batch = mat.n_batch.

- **n_dense** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional(int) the number of dense dimensions in the output matrix. If None, then n_dense = mat.n_dense.

- **on_inefficient** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – optional(string), one of `['error',`` ``'warn',`` ``None]`. Specify the behavior in case of an inefficient reconfiguration. This is defined as a reconfiguration where the size of the resulting representation is much larger than the size of the input representation.

Returns:  
BCOO array  
A BCOO array representing the same sparse array as the input, with the specified layout. `mat_out.todense()` will match `mat.todense()` up to appropriate precision.

Return type:  
mat_out

[](jax.experimental.sparse.bcoo_multiply_sparse.html "previous page")

previous

jax.experimental.sparse.bcoo_multiply_sparse

[](jax.experimental.sparse.bcoo_reduce_sum.html "next page")

next

jax.experimental.sparse.bcoo_reduce_sum

Contents

- [`bcoo_update_layout()`](#jax.experimental.sparse.bcoo_update_layout)

By The JAX authors

© Copyright 2024, The JAX Authors.\
