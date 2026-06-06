- [](../index.html)
- [API Reference](../jax.html)
- jax.copy_to_host_async

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.copy_to_host_async.rst "Download source file")
-  .pdf

# jax.copy_to_host_async

## Contents

- [`copy_to_host_async()`](#jax.copy_to_host_async)

# jax.copy_to_host_async[\#](#jax-copy-to-host-async "Link to this heading")

jax.copy_to_host_async(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L2509-L2533)[\#](#jax.copy_to_host_async "Link to this definition")  
Tries to call a `copy_to_host_async` method on pytree leaves.

For each leaf this method will try to call the `copy_to_host_async` method on the leaf. If the leaf is not a JAX array, or if the leaf does not have a `copy_to_host_async` method, then this method will do nothing to the leaf.

Parameters:  
**x** – a pytree, usually with at least some JAX array instances at its leaves.

Returns:  
A pytree with the same structure and values of the input, where the host copy of the values of all JAX array leaves are started.

[](jax.block_until_ready.html "previous page")

previous

jax.block_until_ready

[](jax.make_mesh.html "next page")

next

jax.make_mesh

Contents

- [`copy_to_host_async()`](#jax.copy_to_host_async)

By The JAX authors

© Copyright 2024, The JAX Authors.\
