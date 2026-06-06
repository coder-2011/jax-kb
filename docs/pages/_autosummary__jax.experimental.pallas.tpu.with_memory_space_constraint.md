- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.with_memory_space_constraint

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.with_memory_space_constraint.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.with_memory_space_constraint

## Contents

- [`with_memory_space_constraint()`](#jax.experimental.pallas.tpu.with_memory_space_constraint)

# jax.experimental.pallas.tpu.with_memory_space_constraint[\#](#jax-experimental-pallas-tpu-with-memory-space-constraint "Link to this heading")

jax.experimental.pallas.tpu.with_memory_space_constraint(*x*, *memory_space*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/primitives.py#L1026-L1058)[\#](#jax.experimental.pallas.tpu.with_memory_space_constraint "Link to this definition")  
Constrains the memory space of an array.

This primitive does not change the value of `x`, but it constrains the memory space where it should be allocated. This is useful to force Pallas to allocate an array in a specific memory space.

As of now, this only operates on the inputs pallas_calls, as in you can apply this to the arguments of a pallas_call and it will constrain them, but other operations will not respect this constraint.

Parameters:  
- **x** ([*jax.Array*](jax.Array.html#jax.Array "jax.Array")) – The array to constrain.

- **memory_space** (*Any*) – The memory space to constrain to.

Returns:  
The array `x` with the memory space constraint.

Return type:  
[jax.Array](jax.Array.html#jax.Array "jax.Array")

[](jax.experimental.pallas.tpu.run_on_first_core.html "previous page")

previous

jax.experimental.pallas.tpu.run_on_first_core

[](../jax.experimental.pallas.mosaic_gpu.html "next page")

next

`jax.experimental.pallas.mosaic_gpu` module

Contents

- [`with_memory_space_constraint()`](#jax.experimental.pallas.tpu.with_memory_space_constraint)

By The JAX authors

© Copyright 2024, The JAX Authors.\
