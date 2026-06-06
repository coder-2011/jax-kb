- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcoo_concatenate

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcoo_concatenate.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcoo_concatenate

## Contents

- [`bcoo_concatenate()`](#jax.experimental.sparse.bcoo_concatenate)

# jax.experimental.sparse.bcoo_concatenate[\#](#jax-experimental-sparse-bcoo-concatenate "Link to this heading")

jax.experimental.sparse.bcoo_concatenate(*operands*, *\**, *dimension*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L1767-L1836)[\#](#jax.experimental.sparse.bcoo_concatenate "Link to this definition")  
Sparse implementation of [`jax.lax.concatenate()`](jax.lax.concatenate.html#jax.lax.concatenate "jax.lax.concatenate")

Parameters:  
- **operands** (*Sequence\[*[*BCOO*](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")*\]*) – Sequence of BCOO arrays to concatenate. The arrays must have equal shapes, except in the dimension axis. Additionally, the arrays must have have equivalent batch, sparse, and dense dimensions.

- **dimension** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Positive integer specifying the dimension along which to concatenate the arrays. The dimension must be among batch or sparse dimensions of the input; concatenation along dense dimensions is not supported.

Returns:  
A BCOO array containing the concatenation of the inputs.

Return type:  
[BCOO](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")

[](jax.experimental.sparse.bcoo_broadcast_in_dim.html "previous page")

previous

jax.experimental.sparse.bcoo_broadcast_in_dim

[](jax.experimental.sparse.bcoo_dot_general.html "next page")

next

jax.experimental.sparse.bcoo_dot_general

Contents

- [`bcoo_concatenate()`](#jax.experimental.sparse.bcoo_concatenate)

By The JAX authors

© Copyright 2024, The JAX Authors.\
