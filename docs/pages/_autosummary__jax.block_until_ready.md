- [](../index.html)
- [API Reference](../jax.html)
- jax.block_until_ready

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.block_until_ready.rst "Download source file")
-  .pdf

# jax.block_until_ready

## Contents

- [`block_until_ready()`](#jax.block_until_ready)

# jax.block_until_ready[\#](#jax-block-until-ready "Link to this heading")

jax.block_until_ready(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L2472-L2508)[\#](#jax.block_until_ready "Link to this definition")  
Tries to call a `block_until_ready` method on pytree leaves.

Parameters:  
**x** – a pytree, usually with at least some JAX array instances at its leaves.

Returns:  
A pytree with the same structure and values of the input, where the values of all JAX array leaves are ready.

[](jax.named_scope.html "previous page")

previous

jax.named_scope

[](jax.copy_to_host_async.html "next page")

next

jax.copy_to_host_async

Contents

- [`block_until_ready()`](#jax.block_until_ready)

By The JAX authors

© Copyright 2024, The JAX Authors.\
