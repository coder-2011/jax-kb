- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.pallas_call

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.pallas_call.rst "Download source file")
-  .pdf

# jax.experimental.pallas.pallas_call

## Contents

- [`pallas_call()`](#jax.experimental.pallas.pallas_call)

# jax.experimental.pallas.pallas_call[\#](#jax-experimental-pallas-pallas-call "Link to this heading")

jax.experimental.pallas.pallas_call(*kernel*, *out_shape*, *\**, *grid_spec=None*, *grid=()*, *in_specs=NoBlockSpec*, *out_specs=NoBlockSpec*, *scratch_shapes=()*, *input_output_aliases={}*, *debug=False*, *interpret=False*, *name=None*, *compiler_params=None*, *cost_estimate=None*, *metadata=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/pallas_call.py#L1138-L1246)[\#](#jax.experimental.pallas.pallas_call "Link to this definition")  
Entry point for creating a Pallas kernel.

In contrast to [`jax.experimental.pallas.kernel()`](jax.experimental.pallas.kernel.html#jax.experimental.pallas.kernel "jax.experimental.pallas.kernel"), this entry point assumes that the kernel will be executed over a `grid`.

See [Pallas Quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html).

Parameters:  
- **kernel** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *None\]*) – the kernel function, that receives a Ref for each input and output. The shape of the Refs are given by the `block_shape` in the corresponding `in_specs` and `out_specs`.

- **out_shape** (*Any*) – a PyTree of [`jax.ShapeDtypeStruct`](jax.ShapeDtypeStruct.html#jax.ShapeDtypeStruct "jax.ShapeDtypeStruct") describing the shape and dtypes of the outputs.

- **grid_spec** (*pallas_core.GridSpec* *\|* *None*) – An alternative way to specify `grid`, `in_specs`, `out_specs` and `scratch_shapes`. If given, those other parameters must not be also given.

- **grid** (*pallas_core.TupleGrid*) – the iteration space, as a tuple of integers. The kernel is executed as many times as `prod(grid)`. See details at [grid, a.k.a. kernels in a loop](../pallas/grid_blockspec.html#pallas-grid).

- **in_specs** (*pallas_core.BlockSpecTree*) – a PyTree of [`jax.experimental.pallas.BlockSpec`](jax.experimental.pallas.BlockSpec.html#jax.experimental.pallas.BlockSpec "jax.experimental.pallas.BlockSpec") with a structure matching that of the positional arguments. The default value for `in_specs` specifies the whole array for all inputs, e.g., as `pl.BlockSpec(x.shape,`` ``lambda`` ``*indices:`` ``(0,)`` ``*`` ``x.ndim)`. See details at [BlockSpec, a.k.a. how to chunk up inputs](../pallas/grid_blockspec.html#pallas-blockspec).

- **out_specs** (*pallas_core.BlockSpecTree*) – a PyTree of [`jax.experimental.pallas.BlockSpec`](jax.experimental.pallas.BlockSpec.html#jax.experimental.pallas.BlockSpec "jax.experimental.pallas.BlockSpec") with a structure matching that of the outputs. The default value for `out_specs` specifies the whole array, e.g., as `pl.BlockSpec(x.shape,`` ``lambda`` ``*indices:`` ``(0,)`` ``*`` ``x.ndim)`. See details at [BlockSpec, a.k.a. how to chunk up inputs](../pallas/grid_blockspec.html#pallas-blockspec).

- **scratch_shapes** (*pallas_core.ScratchShapeTree*) – a PyTree of backend-specific temporary objects required by the kernel, such as temporary buffers, synchronization primitives, etc.

- **input_output_aliases** (*Mapping\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – a dictionary mapping the index of some inputs to the index of the output that aliases them. These indices are in the flattened inputs and outputs (ignoring None values).

- **debug** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, Pallas prints various intermediate forms of the kernel as it is being processed.

- **interpret** (*Any*) – runs the `pallas_call` as a `jax.jit` of a scan over the grid whose body is the kernel lowered as a JAX function. This does not require a TPU or a GPU, and is the only way to run Pallas kernels on CPU. This is useful for debugging.

- **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – if present, specifies the name to use for this kernel call in debugging and error messages. To this name we append the file and line where the kernel function is defined, .e.g: {name} for kernel function {kernel_name} at {file}:{line}. If missing, then we use {kernel_name} at {file}:{line}.

- **compiler_params** (*pallas_core.CompilerParams* *\|* *None*) – Optional compiler parameters. The value should be a backend-specific dataclass ([`jax.experimental.pallas.tpu.CompilerParams`](jax.experimental.pallas.tpu.CompilerParams.html#jax.experimental.pallas.tpu.CompilerParams "jax.experimental.pallas.tpu.CompilerParams"), [`jax.experimental.pallas.triton.CompilerParams`](jax.experimental.pallas.triton.CompilerParams.html#jax.experimental.pallas.triton.CompilerParams "jax.experimental.pallas.triton.CompilerParams"), [`jax.experimental.pallas.mosaic_gpu.CompilerParams`](jax.experimental.pallas.mosaic_gpu.CompilerParams.html#jax.experimental.pallas.mosaic_gpu.CompilerParams "jax.experimental.pallas.mosaic_gpu.CompilerParams")).

- **metadata** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]* *\|* *None*) – Optional dictionary of information about the kernel that will be serialized as JSON in the HLO. Can be used for debugging and analysis.

- **cost_estimate** (*CostEstimate* *\|* *None*)

Returns:  
A function that can be called on a number of positional array arguments to invoke the Pallas kernel.

Return type:  
[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[…, Any\]

[](jax.experimental.pallas.kernel.html "previous page")

previous

jax.experimental.pallas.kernel

[](jax.experimental.pallas.program_id.html "next page")

next

jax.experimental.pallas.program_id

Contents

- [`pallas_call()`](#jax.experimental.pallas.pallas_call)

By The JAX authors

© Copyright 2024, The JAX Authors.\
