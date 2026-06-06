- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.emit_pipeline

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.emit_pipeline.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.emit_pipeline

## Contents

- [`emit_pipeline()`](#jax.experimental.pallas.mosaic_gpu.emit_pipeline)

# jax.experimental.pallas.mosaic_gpu.emit_pipeline[\#](#jax-experimental-pallas-mosaic-gpu-emit-pipeline "Link to this heading")

jax.experimental.pallas.mosaic_gpu.emit_pipeline(*body*, *\**, *grid*, *in_specs=()*, *out_specs=()*, *max_concurrent_steps=1*, *init_carry=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/pipeline.py#L217-L482)[\#](#jax.experimental.pallas.mosaic_gpu.emit_pipeline "Link to this definition")  
Creates a function to emit a manual pipeline within a Pallas kernel.

Parameters:  
- **body** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *T\]*) –

  The pipeline body function, which is called with

  - `indices`: Tuple of current loop indices.

  - `*input_refs`: SMEM refs for inputs.

  - `*output_refs`: SMEM refs for outputs.

  If `init_carry` is provided, `body` receives an additional argument `carry` – the carry from the previous iteration. It must then return the next carry value.

- **grid** (*pallas_core.TupleGrid*) – The grid dimensions for the pipeline.

- **in_specs** (*Sequence\[pallas_core.BlockSpec\]*) – A sequence of [`BlockSpec`](jax.experimental.pallas.BlockSpec.html#jax.experimental.pallas.BlockSpec "jax.experimental.pallas.BlockSpec")s for inputs.

- **out_specs** (*Sequence\[pallas_core.BlockSpec\]*) – A sequence of [`BlockSpec`](jax.experimental.pallas.BlockSpec.html#jax.experimental.pallas.BlockSpec "jax.experimental.pallas.BlockSpec")s for outputs.

- **max_concurrent_steps** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Maximum concurrently active pipeline stages.

- **init_carry** (*T* *\|* *None*) – Optional initial carry. If provided, `body` handles carry-over state between iterations, and the pipeline returns the final carry.

Returns:  
A function that, when called with GMEM input and output refs, executes the pipeline and returns the final carry value (if `init_carry` was used), otherwise it returns None.

[](jax.experimental.pallas.mosaic_gpu.planar_snake.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.planar_snake

[](jax.experimental.pallas.mosaic_gpu.emit_pipeline_warp_specialized.html "next page")

next

jax.experimental.pallas.mosaic_gpu.emit_pipeline_warp_specialized

Contents

- [`emit_pipeline()`](#jax.experimental.pallas.mosaic_gpu.emit_pipeline)

By The JAX authors

© Copyright 2024, The JAX Authors.\
