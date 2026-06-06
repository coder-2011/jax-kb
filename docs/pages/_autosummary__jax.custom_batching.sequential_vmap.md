- [](../index.html)
- [API Reference](../jax.html)
- jax.custom_batching.sequential_vmap

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.custom_batching.sequential_vmap.rst "Download source file")
-  .pdf

# jax.custom_batching.sequential_vmap

## Contents

- [`sequential_vmap()`](#jax.custom_batching.sequential_vmap)

# jax.custom_batching.sequential_vmap[\#](#jax-custom-batching-sequential-vmap "Link to this heading")

jax.custom_batching.sequential_vmap(*f*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/custom_batching.py#L369-L413)[\#](#jax.custom_batching.sequential_vmap "Link to this definition")  
A special case of `custom_vmap` that uses a loop.

A function decorated with `sequential_vmap` will be called sequentially within a loop when batched. This is useful for functions that don’t natively support batch dimensions.

For example:

    >>> @jax.custom_batching.sequential_vmap
    ... def f(x):
    ...   jax.debug.print("{}", x)
    ...   return x + 1
    ...
    >>> jax.vmap(f)(jnp.arange(3))
    0
    1
    2
    Array([1, 2, 3], dtype=int32)

Where the print statements demonstrate that this [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap") is being generated using a loop.

See the documentation for [`custom_vmap`](jax.custom_batching.custom_vmap.html#jax.custom_batching.custom_vmap "jax.custom_batching.custom_vmap") for more details.

[](jax.custom_batching.custom_vmap.def_vmap.html "previous page")

previous

jax.custom_batching.custom_vmap.def_vmap

[](jax.Array.html "next page")

next

jax.Array

Contents

- [`sequential_vmap()`](#jax.custom_batching.sequential_vmap)

By The JAX authors

© Copyright 2024, The JAX Authors.\
