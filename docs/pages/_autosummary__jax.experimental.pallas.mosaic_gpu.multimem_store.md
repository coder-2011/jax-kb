- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.multimem_store

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.multimem_store.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.multimem_store

## Contents

- [`multimem_store()`](#jax.experimental.pallas.mosaic_gpu.multimem_store)

# jax.experimental.pallas.mosaic_gpu.multimem_store[\#](#jax-experimental-pallas-mosaic-gpu-multimem-store "Link to this heading")

jax.experimental.pallas.mosaic_gpu.multimem_store(*source*, *ref*, *collective_axes*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L4778-L4804)[\#](#jax.experimental.pallas.mosaic_gpu.multimem_store "Link to this definition")  
Stores the value to ref on all devices present in collective_axes.

The stores is done using the multimem instructions, meaning that the data is only transferred to the switch once, and broadcasted to all other devices there.

Parameters:  
- **source** ([*jax.Array*](jax.Array.html#jax.Array "jax.Array")) – The value to store.

- **ref** (*\_Ref*) – The GMEM reference to store the value to.

- **collective_axes** (*Hashable* *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Hashable,* *...\]*) – The JAX mesh axes indicating the devices to store to.

[](jax.experimental.pallas.mosaic_gpu.query_cluster_cancel.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.query_cluster_cancel

[](jax.experimental.pallas.mosaic_gpu.multimem_load_reduce.html "next page")

next

jax.experimental.pallas.mosaic_gpu.multimem_load_reduce

Contents

- [`multimem_store()`](#jax.experimental.pallas.mosaic_gpu.multimem_store)

By The JAX authors

© Copyright 2024, The JAX Authors.\
