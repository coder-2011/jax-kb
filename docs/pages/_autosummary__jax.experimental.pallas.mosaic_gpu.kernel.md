- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.kernel

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.kernel.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.kernel

## Contents

- [`kernel()`](#jax.experimental.pallas.mosaic_gpu.kernel)

# jax.experimental.pallas.mosaic_gpu.kernel[\#](#jax-experimental-pallas-mosaic-gpu-kernel "Link to this heading")

jax.experimental.pallas.mosaic_gpu.kernel(*body*, *out_shape*, *\**, *scratch_shapes=()*, *compiler_params=None*, *grid=()*, *grid_names=()*, *cluster=()*, *cluster_names=()*, *num_threads=None*, *thread_name=None*, *interpret=None*, *\*\*mesh_kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/core.py#L242-L368)[\#](#jax.experimental.pallas.mosaic_gpu.kernel "Link to this definition")  
Entry point for defining a Mosaic GPU kernel.

Parameters:  
- **body** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *None\]*) – The kernel body, which should take as arguments the input, output, and scratch Refs. The number of input Refs is determined by the number of arguments passed into kernel returned by this function. The number of output and scratch Refs are determined by out_shape and scratch_shapes respectively.

- **out_shape** ([*object*](https://docs.python.org/3/library/functions.html#object "(in Python v3.14)")) – a PyTree of [`jax.ShapeDtypeStruct`](jax.ShapeDtypeStruct.html#jax.ShapeDtypeStruct "jax.ShapeDtypeStruct") describing the shape and dtypes of the outputs.

- **scratch_shapes** (*pallas_core.ScratchShapeTree*) – an iterable (may be nested) of GPUMemoryRef describing scratch Refs to allocate for this kernel.

- **compiler_params** (*pallas_core.CompilerParams* *\|* *None*) – Additional compiler options. See the CompilerParams dataclass for more details.

- **grid** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*) – A tuple of integers specifying the size of the kernel grid.

- **grid_names** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *...\]*) – The axis names of the grid. Must be the same length as grid.

- **cluster** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*) – A tuple of integers specifying the size of the kernel cluster.

- **cluster_names** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *...\]*) – The axis names of the grid. Must be the same length as cluster.

- **num_threads** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – The number of threads to launch per block. Note that these do not correspond to CUDA threads, but rather to warpgroups on Hopper and Blackwell GPUs.

- **thread_name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – The axis name used to query the thread index.

- **\*\*mesh_kwargs** (*Any*) – Additional mesh kwargs. See Mesh for more details.

- **interpret** (*Any*)

- **\*\*mesh_kwargs**

Returns:  
A function that runs the kernel. It should take any number of input operands and returns an output with the same PyTree structure as out_shape.

[](jax.experimental.pallas.mosaic_gpu.as_torch_kernel.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.as_torch_kernel

[](jax.experimental.pallas.mosaic_gpu.layout_cast.html "next page")

next

jax.experimental.pallas.mosaic_gpu.layout_cast

Contents

- [`kernel()`](#jax.experimental.pallas.mosaic_gpu.kernel)

By The JAX authors

© Copyright 2024, The JAX Authors.\
