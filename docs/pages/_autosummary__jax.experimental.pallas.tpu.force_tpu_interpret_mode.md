- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.force_tpu_interpret_mode

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.force_tpu_interpret_mode.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.force_tpu_interpret_mode

## Contents

- [`force_tpu_interpret_mode()`](#jax.experimental.pallas.tpu.force_tpu_interpret_mode)

# jax.experimental.pallas.tpu.force_tpu_interpret_mode[\#](#jax-experimental-pallas-tpu-force-tpu-interpret-mode "Link to this heading")

jax.experimental.pallas.tpu.force_tpu_interpret_mode(*params=InterpretParams(detect_races=False, out_of_bounds_reads='raise', skip_floating_point_ops=False, uninitialized_memory='nan', num_cores_or_threads=1, vector_clock_size=None, logging_mode=None, dma_execution_mode='on_wait', random_seed=None, grid_point_recorder=None, allow_hbm_allocation_in_run_scoped=False, buffer_bounds='logical')*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L112-L134)[\#](#jax.experimental.pallas.tpu.force_tpu_interpret_mode "Link to this definition")  
Context manager that forces TPU interpret mode under its dynamic context.

TPU interpret mode is a way run Pallas TPU kernels on CPU, while simulating a TPU’s shared memory (HBM, VMEM, etc.), communication (remote and local DMAs), and synchronization operations (semaphores, barriers, etc.). This mode is intended for debugging and testing. See [`InterpretParams`](jax.experimental.pallas.tpu.InterpretParams.html#jax.experimental.pallas.tpu.InterpretParams "jax.experimental.pallas.tpu.InterpretParams") for additional information.

Parameters:  
**params** ([*InterpretParams*](jax.experimental.pallas.tpu.InterpretParams.html#jax.experimental.pallas.tpu.InterpretParams "jax._src.pallas.mosaic.interpret.params.InterpretParams")) – an instance of [`InterpretParams`](jax.experimental.pallas.tpu.InterpretParams.html#jax.experimental.pallas.tpu.InterpretParams "jax.experimental.pallas.tpu.InterpretParams"). Any call to [`jax.experimental.pallas.pallas_call()`](jax.experimental.pallas.pallas_call.html#jax.experimental.pallas.pallas_call "jax.experimental.pallas.pallas_call") or [`jax.experimental.pallas.core_map()`](jax.experimental.pallas.core_map.html#jax.experimental.pallas.core_map "jax.experimental.pallas.core_map") that is traced under this context manager will be run with `interpret=params`. When `params` is not `None`, this will cause those calls to run with TPU interpret mode.

[](jax.experimental.pallas.tpu.to_pallas_key.html "previous page")

previous

jax.experimental.pallas.tpu.to_pallas_key

[](jax.experimental.pallas.tpu.InterpretParams.html "next page")

next

jax.experimental.pallas.tpu.InterpretParams

Contents

- [`force_tpu_interpret_mode()`](#jax.experimental.pallas.tpu.force_tpu_interpret_mode)

By The JAX authors

© Copyright 2024, The JAX Authors.\
