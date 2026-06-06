- [](../index.html)
- [API Reference](../jax.html)
- jax.custom_batching.custom_vmap.def_vmap

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.custom_batching.custom_vmap.def_vmap.rst "Download source file")
-  .pdf

# jax.custom_batching.custom_vmap.def_vmap

## Contents

- [`custom_vmap.def_vmap()`](#jax.custom_batching.custom_vmap.def_vmap)

# jax.custom_batching.custom_vmap.def_vmap[\#](#jax-custom-batching-custom-vmap-def-vmap "Link to this heading")

custom_vmap.def_vmap(*vmap_rule*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/custom_batching.py#L117-L138)[\#](#jax.custom_batching.custom_vmap.def_vmap "Link to this definition")  
Define the vmap rule for this custom_vmap function.

Parameters:  
**vmap_rule** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Any,* *Any\]\]*) – A function that implements the vmap rule. This function should accept the following arguments: (1) an integer `axis_size` as its first argument, (2) a pytree of booleans with the same structure as the inputs to the function, specifying whether each argument is batched, and (3) the batched arguments. It should return a tuple of the batched output and a pytree of booleans with the same structure as the output, specifying whether each output element is batched. See the documentation for [`jax.custom_batching.custom_vmap()`](jax.custom_batching.custom_vmap.html#jax.custom_batching.custom_vmap "jax.custom_batching.custom_vmap") for some examples.

Returns:  
This method passes the rule through, returning `vmap_rule` unchanged.

Return type:  
[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[…, [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[Any, Any\]\]

[](jax.custom_batching.custom_vmap.html "previous page")

previous

jax.custom_batching.custom_vmap

[](jax.custom_batching.sequential_vmap.html "next page")

next

jax.custom_batching.sequential_vmap

Contents

- [`custom_vmap.def_vmap()`](#jax.custom_batching.custom_vmap.def_vmap)

By The JAX authors

© Copyright 2024, The JAX Authors.\
