- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcoo_multiply_dense

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcoo_multiply_dense.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcoo_multiply_dense

## Contents

- [`bcoo_multiply_dense()`](#jax.experimental.sparse.bcoo_multiply_dense)

# jax.experimental.sparse.bcoo_multiply_dense[\#](#jax-experimental-sparse-bcoo-multiply-dense "Link to this heading")

jax.experimental.sparse.bcoo_multiply_dense(*sp_mat*, *v*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L2259-L2270)[\#](#jax.experimental.sparse.bcoo_multiply_dense "Link to this definition")  
An element-wise multiplication between a sparse and a dense array.

Parameters:  
- **lhs** – A BCOO-format array.

- **rhs** – An ndarray.

- **sp_mat** ([*BCOO*](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO"))

- **v** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

Returns:  
An ndarray containing the result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.experimental.sparse.bcoo_gather.html "previous page")

previous

jax.experimental.sparse.bcoo_gather

[](jax.experimental.sparse.bcoo_multiply_sparse.html "next page")

next

jax.experimental.sparse.bcoo_multiply_sparse

Contents

- [`bcoo_multiply_dense()`](#jax.experimental.sparse.bcoo_multiply_dense)

By The JAX authors

© Copyright 2024, The JAX Authors.\
