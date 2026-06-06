- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.async_copy

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.async_copy.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.async_copy

## Contents

- [`async_copy()`](#jax.experimental.pallas.tpu.async_copy)

# jax.experimental.pallas.tpu.async_copy[\#](#jax-experimental-pallas-tpu-async-copy "Link to this heading")

jax.experimental.pallas.tpu.async_copy(*src_ref*, *dst_ref*, *sem*, *\**, *priority=0*, *add=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/primitives.py#L698-L705)[\#](#jax.experimental.pallas.tpu.async_copy "Link to this definition")  
Issues a DMA copying from src_ref to dst_ref.

Parameters:  
- **priority** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **add** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Return type:  
AsyncCopyDescriptor

[](jax.experimental.pallas.tpu.store.html "previous page")

previous

jax.experimental.pallas.tpu.store

[](jax.experimental.pallas.tpu.async_remote_copy.html "next page")

next

jax.experimental.pallas.tpu.async_remote_copy

Contents

- [`async_copy()`](#jax.experimental.pallas.tpu.async_copy)

By The JAX authors

© Copyright 2024, The JAX Authors.\
