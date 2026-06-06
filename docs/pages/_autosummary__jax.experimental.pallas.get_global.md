- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.get_global

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.get_global.rst "Download source file")
-  .pdf

# jax.experimental.pallas.get_global

## Contents

- [`get_global()`](#jax.experimental.pallas.get_global)

# jax.experimental.pallas.get_global[\#](#jax-experimental-pallas-get-global "Link to this heading")

jax.experimental.pallas.get_global(*what*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/primitives.py#L818-L836)[\#](#jax.experimental.pallas.get_global "Link to this definition")  
Returns a global reference that persists across all kernel invocations.

Each call to `get_global` returns a different and unique reference, but one that is stable across invocations of the kernel body.

Parameters:  
**what** (*pallas_core.ScratchShape*) – The reference type to allocate. Each backend has its own set of reference types (e.g., [`jax.experimental.pallas.mosaic_gpu.SemaphoreType`](jax.experimental.pallas.mosaic_gpu.SemaphoreType.html#jax.experimental.pallas.mosaic_gpu.SemaphoreType "jax.experimental.pallas.mosaic_gpu.SemaphoreType") for GPU).

Return type:  
jax_typing.Array

Example:

    sem_ref = pl.get_global(plgpu.SemaphoreType.REGULAR)
    pl.semaphore_signal(sem_ref)
    pl.semaphore_wait(sem_ref)

[](jax.experimental.pallas.dot.html "previous page")

previous

jax.experimental.pallas.dot

[](jax.experimental.pallas.loop.html "next page")

next

jax.experimental.pallas.loop

Contents

- [`get_global()`](#jax.experimental.pallas.get_global)

By The JAX authors

© Copyright 2024, The JAX Authors.\
