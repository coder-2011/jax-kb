- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive

## Contents

- [`tcgen05_commit_arrive()`](#jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive)

# jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive[\#](#jax-experimental-pallas-mosaic-gpu-tcgen05-commit-arrive "Link to this heading")

jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive(*barrier*, *collective_axis=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L2663-L2690)[\#](#jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive "Link to this definition")  
Tracks completion of all preceding `tcgen05_mma` and `async_copy_smem_to_tmem` calls.

Parameters:  
- **barrier** (*\_Ref*) – Barrier Ref for synchronizing with the tensor core. Must have orders_tensor_core set to True.

- **collective_axis** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – The name of the cluster axis along which the operations were performed if it was collective. The cluster axis should have a size of exactly 2, and must be on the minormost cluster axis.

See also

- [`jax.experimental.pallas.mosaic_gpu.tcgen05_mma()`](jax.experimental.pallas.mosaic_gpu.tcgen05_mma.html#jax.experimental.pallas.mosaic_gpu.tcgen05_mma "jax.experimental.pallas.mosaic_gpu.tcgen05_mma")

- `jax.experimental.pallas.mosaic_gpu.async_copy_smem_to_tmem()`

[](jax.experimental.pallas.mosaic_gpu.tcgen05_mma.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.tcgen05_mma

[](jax.experimental.pallas.mosaic_gpu.async_load_tmem.html "next page")

next

jax.experimental.pallas.mosaic_gpu.async_load_tmem

Contents

- [`tcgen05_commit_arrive()`](#jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive)

By The JAX authors

© Copyright 2024, The JAX Authors.\
