- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.get_barrier_semaphore

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.get_barrier_semaphore.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.get_barrier_semaphore

## Contents

- [`get_barrier_semaphore()`](#jax.experimental.pallas.tpu.get_barrier_semaphore)

# jax.experimental.pallas.tpu.get_barrier_semaphore[\#](#jax-experimental-pallas-tpu-get-barrier-semaphore "Link to this heading")

jax.experimental.pallas.tpu.get_barrier_semaphore()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/primitives.py#L774-L796)[\#](#jax.experimental.pallas.tpu.get_barrier_semaphore "Link to this definition")  
Returns a barrier semaphore.

This function returns a barrier semaphore based on the collective_id of the current pallas kernel.

It’s very important that the semaphore is wait-ed back down to 0, or else the semaphores will become corrupted.

It’s also very important that the collective_id is different for each pallas kernel with communication. E.g. if you have two pallas kernels, one that syncs across the X axis of the device mesh and the second that syncs across the Y axis, they must have different collective_ids. However it is legal for two kernels that perform the same synchronization pattern (e.g. only communicating with neighbours on the same mesh axis) to share a collective_id. However, if in doubt, prefer not sharing collective_ids, as doing so incorrectly can lead to silent data corruption or crashes. Note that reusing the same collective_id doesn’t guarantee that the same semaphore is provided by XLA.

[](jax.experimental.pallas.tpu.core_barrier.html "previous page")

previous

jax.experimental.pallas.tpu.core_barrier

[](jax.experimental.pallas.tpu.get_tpu_info.html "next page")

next

jax.experimental.pallas.tpu.get_tpu_info

Contents

- [`get_barrier_semaphore()`](#jax.experimental.pallas.tpu.get_barrier_semaphore)

By The JAX authors

© Copyright 2024, The JAX Authors.\
