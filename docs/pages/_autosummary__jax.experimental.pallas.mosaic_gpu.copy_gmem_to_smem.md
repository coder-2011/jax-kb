- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem

## Contents

- [`copy_gmem_to_smem()`](#jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem)

# jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem[\#](#jax-experimental-pallas-mosaic-gpu-copy-gmem-to-smem "Link to this heading")

jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem(*src*, *dst*, *barrier*, *\**, *collective_axes=None*, *leader_tracked=None*, *oob_mode=OOBFillMode.ZEROS*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L950-L1039)[\#](#jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem "Link to this definition")  
Asynchronously copies a GMEM reference to a SMEM reference.

If collective_axes is specified, this performs a multicast copy where all CUDA blocks that share the same index along the collective axis receive a copy of the same block of data loaded from dst to src.

If both `collective_axes` and `leader_tracked` are specified as `CopyPartition.PARTITIONED(axis)`, this will perform a partitioned collective copy where each block in the cluster will receive a tile of `transfer_size`` ``//`` ``cluster_size` data from the `src` Ref. For example, if `src` has a shape of (256, 256) and a partitioned copy is performed along axis 0 with cluster size 2, then the first block will receive `src[0:128,`` ``:]` and the second will receive `src[128:256,`` ``:]`.

If both `collective_axes` and `leader_tracked` are specified as `CopyPartition.REPLICATED`, this will perform a replicated copy where all blocks load the same data but only the first block in the collective tracks progress via barrier arrivals.

NOTE: Only the first block in the cluster will arrive on the barrier, and an additional cluster barrier is necessary to ensure that all blocks in the cluster have finished the copy.

Parameters:  
- **src** (*\_Ref*) – The source Ref. Must be in GMEM.

- **dst** (*\_Ref*) – The destination Ref. Must be in SMEM.

- **barrier** (*\_Ref*) – The barrier to use for tracking completion of the copy.

- **collective_axes** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *...\]* *\|* *None*) – The collective axes to use for the copy.

- **leader_tracked** (*CopyPartition* *\|* *None*) – If specified, only the leader block in the cluster will observe the completion of the copy. If `CopyPartition.PARTITIONED(axis)`, performs a partitioned collective copy along the given axis. If `CopyPartition.REPLICATED`, all blocks load the same data.

- **oob_mode** (*OOBFillMode*) – The optional out-of-bounds fill mode. Can be `OOBFillMode.UNDEFINED`, `OOBFillMode.PROMISE_IN_BOUNDS` or `OOBFillMode.ZEROS`.

Return type:  
None

See also

[`jax.experimental.pallas.mosaic_gpu.barrier_arrive()`](jax.experimental.pallas.mosaic_gpu.barrier_arrive.html#jax.experimental.pallas.mosaic_gpu.barrier_arrive "jax.experimental.pallas.mosaic_gpu.barrier_arrive") [`jax.experimental.pallas.mosaic_gpu.barrier_wait()`](jax.experimental.pallas.mosaic_gpu.barrier_wait.html#jax.experimental.pallas.mosaic_gpu.barrier_wait "jax.experimental.pallas.mosaic_gpu.barrier_wait")

[](jax.experimental.pallas.mosaic_gpu.commit_smem.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.commit_smem

[](jax.experimental.pallas.mosaic_gpu.copy_smem_to_gmem.html "next page")

next

jax.experimental.pallas.mosaic_gpu.copy_smem_to_gmem

Contents

- [`copy_gmem_to_smem()`](#jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem)

By The JAX authors

© Copyright 2024, The JAX Authors.\
