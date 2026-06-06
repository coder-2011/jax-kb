- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.emit_pipeline_with_allocations

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.emit_pipeline_with_allocations.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.emit_pipeline_with_allocations

## Contents

- [`emit_pipeline_with_allocations()`](#jax.experimental.pallas.tpu.emit_pipeline_with_allocations)

# jax.experimental.pallas.tpu.emit_pipeline_with_allocations[\#](#jax-experimental-pallas-tpu-emit-pipeline-with-allocations "Link to this heading")

jax.experimental.pallas.tpu.emit_pipeline_with_allocations(*body*, *\**, *grid*, *in_specs=()*, *out_specs=()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/pipeline.py#L1781-L1813)[\#](#jax.experimental.pallas.tpu.emit_pipeline_with_allocations "Link to this definition")  
Creates pallas pipeline and top-level allocation preparation functions.

Parameters:  
- **body** – pallas kernel to set up pipeline for.

- **grid** – a pallas grid definition.

- **in_specs** – input pallas block specs

- **out_specs** – output pallas block specs

Returns:  
(emit_pipeline, make_allocations) function pair, where  
- emit_pipeline is the pallas pipeline function.

- make_allocations is a function to create buffered refs for the inner pipeline that can be created at the top-level of a pallas call to be reused across multiple invocations of the inner pipeline.

[](jax.experimental.pallas.tpu.emit_pipeline.html "previous page")

previous

jax.experimental.pallas.tpu.emit_pipeline

[](jax.experimental.pallas.tpu.prng_seed.html "next page")

next

jax.experimental.pallas.tpu.prng_seed

Contents

- [`emit_pipeline_with_allocations()`](#jax.experimental.pallas.tpu.emit_pipeline_with_allocations)

By The JAX authors

© Copyright 2024, The JAX Authors.\
