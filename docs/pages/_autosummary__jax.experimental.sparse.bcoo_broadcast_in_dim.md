- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcoo_broadcast_in_dim

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcoo_broadcast_in_dim.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcoo_broadcast_in_dim

## Contents

- [`bcoo_broadcast_in_dim()`](#jax.experimental.sparse.bcoo_broadcast_in_dim)

# jax.experimental.sparse.bcoo_broadcast_in_dim[\#](#jax-experimental-sparse-bcoo-broadcast-in-dim "Link to this heading")

jax.experimental.sparse.bcoo_broadcast_in_dim(*mat*, *\**, *shape*, *broadcast_dimensions*, *sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L1707-L1726)[\#](#jax.experimental.sparse.bcoo_broadcast_in_dim "Link to this definition")  
Expand the size and rank of a BCOO array by duplicating the data.

A BCOO equivalence to jax.lax.broadcast_in_dim.

Parameters:  
- **mat** ([*BCOO*](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")) – A BCOO-format array.

- **shape** (*Shape*) – The shape of the target array.

- **broadcast_dimensions** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – The dimension in the shape of the target array which each dimension of the operand (`mat`) shape corresponds to.

Returns:  
A BCOO-format array containing the target array.

Return type:  
[BCOO](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")

[](jax.experimental.sparse.BCOO.html "previous page")

previous

jax.experimental.sparse.BCOO

[](jax.experimental.sparse.bcoo_concatenate.html "next page")

next

jax.experimental.sparse.bcoo_concatenate

Contents

- [`bcoo_broadcast_in_dim()`](#jax.experimental.sparse.bcoo_broadcast_in_dim)

By The JAX authors

© Copyright 2024, The JAX Authors.\
