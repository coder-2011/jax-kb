- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcoo_multiply_sparse

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcoo_multiply_sparse.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcoo_multiply_sparse

## Contents

- [`bcoo_multiply_sparse()`](#jax.experimental.sparse.bcoo_multiply_sparse)

# jax.experimental.sparse.bcoo_multiply_sparse[\#](#jax-experimental-sparse-bcoo-multiply-sparse "Link to this heading")

jax.experimental.sparse.bcoo_multiply_sparse(*lhs*, *rhs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L2192-L2206)[\#](#jax.experimental.sparse.bcoo_multiply_sparse "Link to this definition")  
An element-wise multiplication of two sparse arrays.

Parameters:  
- **lhs** ([*BCOO*](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")) – A BCOO-format array.

- **rhs** ([*BCOO*](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")) – A BCOO-format array.

Returns:  
An BCOO-format array containing the result.

Return type:  
[BCOO](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")

[](jax.experimental.sparse.bcoo_multiply_dense.html "previous page")

previous

jax.experimental.sparse.bcoo_multiply_dense

[](jax.experimental.sparse.bcoo_update_layout.html "next page")

next

jax.experimental.sparse.bcoo_update_layout

Contents

- [`bcoo_multiply_sparse()`](#jax.experimental.sparse.bcoo_multiply_sparse)

By The JAX authors

© Copyright 2024, The JAX Authors.\
