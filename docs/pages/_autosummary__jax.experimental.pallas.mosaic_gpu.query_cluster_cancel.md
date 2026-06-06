- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.query_cluster_cancel

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.query_cluster_cancel.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.query_cluster_cancel

## Contents

- [`query_cluster_cancel()`](#jax.experimental.pallas.mosaic_gpu.query_cluster_cancel)

# jax.experimental.pallas.mosaic_gpu.query_cluster_cancel[\#](#jax-experimental-pallas-mosaic-gpu-query-cluster-cancel "Link to this heading")

jax.experimental.pallas.mosaic_gpu.query_cluster_cancel(*result_ref*, *grid_names*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L4489-L4523)[\#](#jax.experimental.pallas.mosaic_gpu.query_cluster_cancel "Link to this definition")  
Decodes the result of a `try_cluster_cancel` operation.

It interprets the 16-byte opaque response written to shared memory by a completed `try_cluster_cancel` call to determine if a new work unit was successfully claimed.

Parameters:  
- **result_ref** (*\_Ref*) – The SMEM ref containing the query response.

- **grid_names** (*Sequence\[Hashable\]*) – A tuple of grid axis names to query for.

Returns:  
- the grid indices for the requested axis names.

- A boolean indicating if the cancellation was successful.

Return type:  
A tuple containing the decoded response

See also

[`jax.experimental.pallas.mosaic_gpu.try_cluster_cancel()`](jax.experimental.pallas.mosaic_gpu.try_cluster_cancel.html#jax.experimental.pallas.mosaic_gpu.try_cluster_cancel "jax.experimental.pallas.mosaic_gpu.try_cluster_cancel")

[](jax.experimental.pallas.mosaic_gpu.try_cluster_cancel.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.try_cluster_cancel

[](jax.experimental.pallas.mosaic_gpu.multimem_store.html "next page")

next

jax.experimental.pallas.mosaic_gpu.multimem_store

Contents

- [`query_cluster_cancel()`](#jax.experimental.pallas.mosaic_gpu.query_cluster_cancel)

By The JAX authors

© Copyright 2024, The JAX Authors.\
