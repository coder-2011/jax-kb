- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.multimem_load_reduce

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.multimem_load_reduce.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.multimem_load_reduce

## Contents

- [`multimem_load_reduce()`](#jax.experimental.pallas.mosaic_gpu.multimem_load_reduce)

# jax.experimental.pallas.mosaic_gpu.multimem_load_reduce[\#](#jax-experimental-pallas-mosaic-gpu-multimem-load-reduce "Link to this heading")

jax.experimental.pallas.mosaic_gpu.multimem_load_reduce(*ref*, *\**, *collective_axes*, *reduction_op*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L4986-L5019)[\#](#jax.experimental.pallas.mosaic_gpu.multimem_load_reduce "Link to this definition")  
Loads from a GMEM reference on all devices present in collective_axes and reduces the loaded values.

The supported dtypes are: `jnp.float32`, `jnp.float16`, `jnp.bfloat16`, `jnp.float8_e5m2`, `jnp.float8_e4m3fn`, `jnp.int32` and `jnp.int64`.

8-bit floating point dtypes are only supported on Blackwell GPUs.

Parameters:  
- **ref** (*\_Ref*) – The GMEM reference to load from.

- **collective_axes** (*Hashable* *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Hashable,* *...\]*) – The JAX mesh axes indicating the devices to load from.

- **reduction_op** (*mgpu.MultimemReductionOp*) – The reduction operation to perform on the loaded values. The allowed values are add (all dtypes), min, max (all dtypes but f32), as well as and, or and xor (integer types only).

Return type:  
[jax.Array](jax.Array.html#jax.Array "jax.Array")

[](jax.experimental.pallas.mosaic_gpu.multimem_store.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.multimem_store

[](jax.experimental.pallas.mosaic_gpu.ACC.html "next page")

next

jax.experimental.pallas.mosaic_gpu.ACC

Contents

- [`multimem_load_reduce()`](#jax.experimental.pallas.mosaic_gpu.multimem_load_reduce)

By The JAX authors

© Copyright 2024, The JAX Authors.\
