- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.try_cluster_cancel

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.try_cluster_cancel.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.try_cluster_cancel

## Contents

- [`try_cluster_cancel()`](#jax.experimental.pallas.mosaic_gpu.try_cluster_cancel)

# jax.experimental.pallas.mosaic_gpu.try_cluster_cancel[\#](#jax-experimental-pallas-mosaic-gpu-try-cluster-cancel "Link to this heading")

jax.experimental.pallas.mosaic_gpu.try_cluster_cancel(*result_ref*, *barrier*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L4383-L4423)[\#](#jax.experimental.pallas.mosaic_gpu.try_cluster_cancel "Link to this definition")  
Initiates an async request to claim a new work unit from the grid.

It allows an SM to dynamically acquire work by atomically canceling the launch of a pending cluster from the grid and claiming its CTA ID as the next unit of work.

Note that this operation must be called collectively by all Pallas threads.

Parameters:  
- **result_ref** (*\_Ref*) – An SMEM ref where the 16-byte result will be stored.

- **barrier** (*\_Ref*) – A barrier used to coordinate the completion of the query.

Return type:  
None

See also

[`jax.experimental.pallas.mosaic_gpu.query_cluster_cancel()`](jax.experimental.pallas.mosaic_gpu.query_cluster_cancel.html#jax.experimental.pallas.mosaic_gpu.query_cluster_cancel "jax.experimental.pallas.mosaic_gpu.query_cluster_cancel")

[](jax.experimental.pallas.mosaic_gpu.commit_tmem.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.commit_tmem

[](jax.experimental.pallas.mosaic_gpu.query_cluster_cancel.html "next page")

next

jax.experimental.pallas.mosaic_gpu.query_cluster_cancel

Contents

- [`try_cluster_cancel()`](#jax.experimental.pallas.mosaic_gpu.try_cluster_cancel)

By The JAX authors

© Copyright 2024, The JAX Authors.\
