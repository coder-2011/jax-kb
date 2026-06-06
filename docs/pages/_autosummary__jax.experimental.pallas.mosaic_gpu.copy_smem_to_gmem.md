- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.copy_smem_to_gmem

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.copy_smem_to_gmem.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.copy_smem_to_gmem

## Contents

- [`copy_smem_to_gmem()`](#jax.experimental.pallas.mosaic_gpu.copy_smem_to_gmem)

# jax.experimental.pallas.mosaic_gpu.copy_smem_to_gmem[\#](#jax-experimental-pallas-mosaic-gpu-copy-smem-to-gmem "Link to this heading")

jax.experimental.pallas.mosaic_gpu.copy_smem_to_gmem(*src*, *dst*, *predicate=None*, *\**, *commit_group=True*, *reduction_op=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L421-L472)[\#](#jax.experimental.pallas.mosaic_gpu.copy_smem_to_gmem "Link to this definition")  
Asynchronously copies a SMEM reference to a GMEM reference.

Parameters:  
- **src** (*\_Ref*) – The SMEM reference to copy from.

- **dst** (*\_Ref*) – The GMEM reference to copy to.

- **predicate** ([*jax.Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*) – A boolean indicating whether the copy should be performed. If `None`, the copy is always performed.

- **commit_group** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If `True`, this and any previously uncommitted copies are committed to a group and can be awaited jointly via [`jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem()`](jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem.html#jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem "jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem").

- **reduction_op** (*mgpu.TMAReductionOp* *\|* *None*) – If set, perform the specified reduction operation when storing to GMEM. For example, using `"add"` is conceptually equivalent to doing `src`` ``+=`` ``dst`.

Return type:  
None

See also

[`jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem()`](jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem.html#jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem "jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem") [`jax.experimental.pallas.mosaic_gpu.commit_smem()`](jax.experimental.pallas.mosaic_gpu.commit_smem.html#jax.experimental.pallas.mosaic_gpu.commit_smem "jax.experimental.pallas.mosaic_gpu.commit_smem")

[](jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem

[](jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem.html "next page")

next

jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem

Contents

- [`copy_smem_to_gmem()`](#jax.experimental.pallas.mosaic_gpu.copy_smem_to_gmem)

By The JAX authors

© Copyright 2024, The JAX Authors.\
