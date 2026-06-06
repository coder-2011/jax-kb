- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.make_async_copy

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.make_async_copy.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.make_async_copy

## Contents

- [`make_async_copy()`](#jax.experimental.pallas.tpu.make_async_copy)

# jax.experimental.pallas.tpu.make_async_copy[\#](#jax-experimental-pallas-tpu-make-async-copy "Link to this heading")

jax.experimental.pallas.tpu.make_async_copy(*src_ref*, *dst_ref*, *sem*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/primitives.py#L677-L696)[\#](#jax.experimental.pallas.tpu.make_async_copy "Link to this definition")  
Creates a description of an asynchronous copy operation.

Parameters:  
- **src_ref** – The source Reference.

- **dst_ref** – The destination Reference.

- **sem** – The semaphore used to track completion of the copy.

Returns:  
An AsyncCopyDescriptor.

Return type:  
AsyncCopyDescriptor

[](jax.experimental.pallas.tpu.async_remote_copy.html "previous page")

previous

jax.experimental.pallas.tpu.async_remote_copy

[](jax.experimental.pallas.tpu.make_async_remote_copy.html "next page")

next

jax.experimental.pallas.tpu.make_async_remote_copy

Contents

- [`make_async_copy()`](#jax.experimental.pallas.tpu.make_async_copy)

By The JAX authors

© Copyright 2024, The JAX Authors.\
