- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.semaphore_signal_parallel

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.semaphore_signal_parallel.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.semaphore_signal_parallel

## Contents

- [`semaphore_signal_parallel()`](#jax.experimental.pallas.mosaic_gpu.semaphore_signal_parallel)

# jax.experimental.pallas.mosaic_gpu.semaphore_signal_parallel[\#](#jax-experimental-pallas-mosaic-gpu-semaphore-signal-parallel "Link to this heading")

jax.experimental.pallas.mosaic_gpu.semaphore_signal_parallel(*\*signals*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L4176-L4206)[\#](#jax.experimental.pallas.mosaic_gpu.semaphore_signal_parallel "Link to this definition")  
Signals multiple semaphores without any guaranteed ordering of signal arrivals.

This primitive is largely equivalent to:

    for sem in semaphores:
      pl.semaphore_signal(sem, inc, device_id=device_id)

only unlike the loop above, it does not guarantee any ordering of signal arrivals. In particular, the target device might observe a signal on `semaphores[1]` before it observes a signal on `semaphores[0]`. This operation still guarantees that any side effects performed before the signal will be fully performed and visible before any of the signals arrive.

The relaxed requirements make the whole operation significantly cheaper on GPUs, as a single expensive memory fence can be used for all signals (instead of an expensive fence for each signal).

Parameters:  
**signals** ([*SemaphoreSignal*](jax.experimental.pallas.mosaic_gpu.SemaphoreSignal.html#jax.experimental.pallas.mosaic_gpu.SemaphoreSignal "jax.experimental.pallas.mosaic_gpu.SemaphoreSignal"))

[](jax.experimental.pallas.mosaic_gpu.barrier_wait.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.barrier_wait

[](jax.experimental.pallas.mosaic_gpu.SemaphoreSignal.html "next page")

next

jax.experimental.pallas.mosaic_gpu.SemaphoreSignal

Contents

- [`semaphore_signal_parallel()`](#jax.experimental.pallas.mosaic_gpu.semaphore_signal_parallel)

By The JAX authors

© Copyright 2024, The JAX Authors.\
