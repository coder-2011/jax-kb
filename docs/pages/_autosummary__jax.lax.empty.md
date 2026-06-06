- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.empty

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.empty.rst "Download source file")
-  .pdf

# jax.lax.empty

## Contents

- [`empty()`](#jax.lax.empty)

# jax.lax.empty[\#](#jax-lax-empty "Link to this heading")

jax.lax.empty(*shape*, *dtype*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L9598-L9621)[\#](#jax.lax.empty "Link to this definition")  
Create an empty array of possibly uninitialized values.

This initialization is backend dependent.

Parameters:  
- **shape** – int or sequence of ints specifying the shape of the created array.

- **dtype** – dtype for the created array.

- **out_sharding** – (optional) [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") or `NamedSharding` representing the sharding of the created array (see [explicit sharding](https://docs.jax.dev/en/latest/parallel.html) for more details).

Returns:  
Uninitialized array of the specified shape, dtype, and sharding.

Examples

    >>> lax.empty(3, jnp.float32)  
    Array([-5.7326739e+29 -7.7323739e+29 -3.14159256e-29], dtype=float32)

[](jax.lax.dynamic_update_slice_in_dim.html "previous page")

previous

jax.lax.dynamic_update_slice_in_dim

[](jax.lax.eq.html "next page")

next

jax.lax.eq

Contents

- [`empty()`](#jax.lax.empty)

By The JAX authors

© Copyright 2024, The JAX Authors.\
