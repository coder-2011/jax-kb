- [](../index.html)
- [API Reference](../jax.html)
- jax.array_garbage_collection_guard

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.array_garbage_collection_guard.rst "Download source file")
-  .pdf

# jax.array_garbage_collection_guard

## Contents

- [`array_garbage_collection_guard`](#jax.array_garbage_collection_guard)

# jax.array_garbage_collection_guard[\#](#jax-array-garbage-collection-guard "Link to this heading")

jax.array_garbage_collection_guard *= \<jax.\_src.config.State object\>*[\#](#jax.array_garbage_collection_guard "Link to this definition")  
Context manager for jax_array_garbage_collection_guard config option.

Select garbage collection guard level for `jax.Array` objects.

This option can be used to control what happens when a `jax.Array` object is garbage collected. It is desirable for `jax.Array` objects to be freed by Python reference counting rather than garbage collection in order to avoid device memory being held by the arrays until garbage collection occurs.

Valid values are:

- `allow`: do not log garbage collection of `jax.Array` objects.

- `log`: log an error when a `jax.Array` is garbage collected.

- `fatal`: fatal error if a `jax.Array` is garbage collected.

Default is `allow`. Note that not all cycles may be detected.

Parameters:  
**new_val** (*Any*)

[](jax.experimental.sparse.linalg.lobpcg_standard.html "previous page")

previous

jax.experimental.sparse.linalg.lobpcg_standard

[](jax.config.html "next page")

next

jax.config

Contents

- [`array_garbage_collection_guard`](#jax.array_garbage_collection_guard)

By The JAX authors

© Copyright 2024, The JAX Authors.\
