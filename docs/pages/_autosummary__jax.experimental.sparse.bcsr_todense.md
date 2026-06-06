- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcsr_todense

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcsr_todense.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcsr_todense

## Contents

- [`bcsr_todense()`](#jax.experimental.sparse.bcsr_todense)

# jax.experimental.sparse.bcsr_todense[\#](#jax-experimental-sparse-bcsr-todense "Link to this heading")

jax.experimental.sparse.bcsr_todense(*mat*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcsr.py#L311-L321)[\#](#jax.experimental.sparse.bcsr_todense "Link to this definition")  
Convert batched sparse matrix to a dense matrix.

Parameters:  
**mat** ([*BCSR*](jax.experimental.sparse.BCSR.html#jax.experimental.sparse.BCSR "jax.experimental.sparse.BCSR")) – BCSR matrix.

Returns:  
The dense version of `mat`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.experimental.sparse.bcsr_fromdense.html "previous page")

previous

jax.experimental.sparse.bcsr_fromdense

[](jax.experimental.sparse.COO.html "next page")

next

jax.experimental.sparse.COO

Contents

- [`bcsr_todense()`](#jax.experimental.sparse.bcsr_todense)

By The JAX authors

© Copyright 2024, The JAX Authors.\
