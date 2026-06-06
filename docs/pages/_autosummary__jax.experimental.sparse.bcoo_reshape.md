- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcoo_reshape

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcoo_reshape.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcoo_reshape

## Contents

- [`bcoo_reshape()`](#jax.experimental.sparse.bcoo_reshape)

# jax.experimental.sparse.bcoo_reshape[\#](#jax-experimental-sparse-bcoo-reshape "Link to this heading")

jax.experimental.sparse.bcoo_reshape(*mat*, *\**, *new_sizes*, *dimensions=None*, *sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L1838-L1901)[\#](#jax.experimental.sparse.bcoo_reshape "Link to this definition")  
Sparse implementation of [`jax.lax.reshape()`](jax.lax.reshape.html#jax.lax.reshape "jax.lax.reshape").

Parameters:  
- **operand** – BCOO array to be reshaped.

- **new_sizes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – sequence of integers specifying the resulting shape. The size of the final array must match the size of the input. This must be specified such that batch, sparse, and dense dimensions do not mix.

- **dimensions** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – optional sequence of integers specifying the permutation order of the input shape. If specified, the length must match `operand.shape`. Additionally, dimensions must only permute among like dimensions of mat: batch, sparse, and dense dimensions cannot be permuted.

- **mat** ([*BCOO*](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO"))

Returns:  
reshaped array.

Return type:  
out

[](jax.experimental.sparse.bcoo_reduce_sum.html "previous page")

previous

jax.experimental.sparse.bcoo_reduce_sum

[](jax.experimental.sparse.bcoo_slice.html "next page")

next

jax.experimental.sparse.bcoo_slice

Contents

- [`bcoo_reshape()`](#jax.experimental.sparse.bcoo_reshape)

By The JAX authors

© Copyright 2024, The JAX Authors.\
