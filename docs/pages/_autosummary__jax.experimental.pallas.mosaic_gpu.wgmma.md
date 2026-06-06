- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.wgmma

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.wgmma.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.wgmma

## Contents

- [`wgmma()`](#jax.experimental.pallas.mosaic_gpu.wgmma)

# jax.experimental.pallas.mosaic_gpu.wgmma[\#](#jax-experimental-pallas-mosaic-gpu-wgmma "Link to this heading")

jax.experimental.pallas.mosaic_gpu.wgmma(*acc*, *a*, *b*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L1495-L1554)[\#](#jax.experimental.pallas.mosaic_gpu.wgmma "Link to this definition")  
Performs an asynchronous warp group matmul-accumulate on the given references.

Conceptually, this is equivalent to doing `acc[...]`` ``+=`` ``a[...]`` ``@`` ``b[...]`, except that the computation is performed asynchronously.

Parameters:  
- **acc** (*gpu_core.WGMMAAbstractAccumulatorRef*) – The accumulator reference. Needs to be allocated via [`jax.experimental.pallas.run_scoped()`](jax.experimental.pallas.run_scoped.html#jax.experimental.pallas.run_scoped "jax.experimental.pallas.run_scoped") called with a [`jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef()`](jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef.html#jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef "jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef").

- **a** – The left hand side operand reference.

- **b** – The right hand side operand reference.

Return type:  
None

See also

[`jax.experimental.pallas.mosaic_gpu.wgmma_wait()`](jax.experimental.pallas.mosaic_gpu.wgmma_wait.html#jax.experimental.pallas.mosaic_gpu.wgmma_wait "jax.experimental.pallas.mosaic_gpu.wgmma_wait")

[](jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem

[](jax.experimental.pallas.mosaic_gpu.wgmma_wait.html "next page")

next

jax.experimental.pallas.mosaic_gpu.wgmma_wait

Contents

- [`wgmma()`](#jax.experimental.pallas.mosaic_gpu.wgmma)

By The JAX authors

© Copyright 2024, The JAX Authors.\
