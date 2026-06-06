- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.emit_pipeline_warp_specialized

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.emit_pipeline_warp_specialized.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.emit_pipeline_warp_specialized

## Contents

- [`emit_pipeline_warp_specialized()`](#jax.experimental.pallas.mosaic_gpu.emit_pipeline_warp_specialized)

# jax.experimental.pallas.mosaic_gpu.emit_pipeline_warp_specialized[\#](#jax-experimental-pallas-mosaic-gpu-emit-pipeline-warp-specialized "Link to this heading")

jax.experimental.pallas.mosaic_gpu.emit_pipeline_warp_specialized(*body*, *\**, *grid*, *memory_registers*, *in_specs=()*, *out_specs=()*, *max_concurrent_steps=2*, *wg_axis*, *num_compute_wgs*, *pipeline_state=None*, *manual_consumed_barriers=False*, *compute_context=None*, *memory_thread_idx=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/pipeline.py#L529-L1060)[\#](#jax.experimental.pallas.mosaic_gpu.emit_pipeline_warp_specialized "Link to this definition")  
Creates a function to emit a warp-specialized pipeline.

The `body` function should have the following signature (without carry). `consumed_barriers` is an optional argument that is only passed if the `manual_consumed_barriers` argument is True:

    def body(indices, *input_refs, *output_refs, *consumed_barriers) -> None:

or with a carries enabled (enabled via the `compute_context` argument), where the body returns the next carry:

    def body(
        indices, *input_refs, *output_refs, *consumed_barriers, carry
    ) -> Carry:

When `manual_consumed_barriers` is True, the user must arrive on all the consumed barriers from all compute warpgroups at each pipeline step.

Parameters:  
- **body** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *None\]*) – The pipeline body.

- **grid** (*pallas_core.TupleGrid*) – The grid to use for the pipeline.

- **memory_registers** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The number of registers to reserve for the memory thread. For H100 GPUs, 40 is a reasonable value.

- **in_specs** (*BlockSpecPytree*) – The block specs for the inputs.

- **out_specs** (*BlockSpecPytree*) – The block specs for the outputs.

- **max_concurrent_steps** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The maximum number of sequential stages that are active concurrently. Defaults to 2.

- **wg_axis** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The axis name for the warp group axis.

- **num_compute_wgs** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The number of compute warpgroups

- **manual_consumed_barriers** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, consumed barriers will be passed into the body function after the output refs. There will be one barrier per input and will be passed in the same order.

- **compute_context** (*ComputeContext* *\|* *None*) – If specified, enables carries in the pipeline and allows a user-specified prologue/epilogue that is only executed in the compute thread. The signature of the pipeline body function will be modified such that the last argument will be the current carry and it must return the next carry. The compute_context itself should follow the signature of ComputeContext and take a pipeline function as its sole argument. Calling the pipeline with the initial carry will run the pipeline and return the final carry.

- **memory_thread_idx** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – The index of the memory thread. If not specified, defaults to the last thread.

- **pipeline_state** ([*jax.Array*](jax.Array.html#jax.Array "jax.Array") *\|* *PipelinePipeline* *\|* *None*) –

  If multiple pipelines that have almost the same parameters (only in/out_specs and body can differ) are going to be evaluated in sequence, this argument can be used to avoid pipeline bubbles between their invocations. The first pipeline in the sequence should use the `START` state, followed by an arbitrary number of `STEADY` states, followed by a single `STOP` state. Note that until the pipeline with `STOP` is done, the memory thread will not wait for the compute threads to complete and fully consume their work. Any modification of their operands other than invoking another pipeline is disallowed.

  Important: To achieve bubble-free execution, it is important to also use the manual allocation mode by calling `get_allocations` on the returned function, passing the result to `pl.run_scoped` and the provided results to the returned function as an `allocations` keyword argument. Otherwise, the pipeline function will perform the scoped allocation itself which can lead to synchronization that can still cause pipeline bubbles.

Return type:  
WarpSpecializedPipeline

[](jax.experimental.pallas.mosaic_gpu.emit_pipeline.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.emit_pipeline

[](jax.experimental.pallas.mosaic_gpu.nd_loop.html "next page")

next

jax.experimental.pallas.mosaic_gpu.nd_loop

Contents

- [`emit_pipeline_warp_specialized()`](#jax.experimental.pallas.mosaic_gpu.emit_pipeline_warp_specialized)

By The JAX authors

© Copyright 2024, The JAX Authors.\
