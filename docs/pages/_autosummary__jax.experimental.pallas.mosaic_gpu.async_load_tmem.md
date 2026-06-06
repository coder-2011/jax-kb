- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.async_load_tmem

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.async_load_tmem.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.async_load_tmem

## Contents

- [`async_load_tmem()`](#jax.experimental.pallas.mosaic_gpu.async_load_tmem)

# jax.experimental.pallas.mosaic_gpu.async_load_tmem[\#](#jax-experimental-pallas-mosaic-gpu-async-load-tmem "Link to this heading")

jax.experimental.pallas.mosaic_gpu.async_load_tmem(*src*, *\**, *layout=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L3568-L3605)[\#](#jax.experimental.pallas.mosaic_gpu.async_load_tmem "Link to this definition")  
Performs an asynchronous load from the TMEM array.

The load operation is only partly asynchronous. The returned array can be used immediately, without any additional synchronization. However, it cannot be assumed that the read from TMEM has completed when the function returns. If you ever attempt to overwrite the read region, you should ensure that `wait_load_tmem` has been called before that happens. Failure to do so can result in nondeterministic data races.

For example, the following sequence of operations at the end of the kernel is valid, even though the TMEM load is never awaited:

    smem_ref[...] = plgpu.async_load_tmem(tmem_ref)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(smem_ref, gmem_ref)
    plgpu.wait_smem_to_gmem(0)

However, if the kernel was persistent and might reuse the TMEM again, the sequence should be extended with a call to `wait_load_tmem`.

Parameters:  
- **src** (*\_Ref*) – The TMEM reference to load from.

- **layout** (*SomeLayout* *\|* *None*) – The optional layout hint to use for the resulting array.

Return type:  
[jax.Array](jax.Array.html#jax.Array "jax.Array")

[](jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive

[](jax.experimental.pallas.mosaic_gpu.async_store_tmem.html "next page")

next

jax.experimental.pallas.mosaic_gpu.async_store_tmem

Contents

- [`async_load_tmem()`](#jax.experimental.pallas.mosaic_gpu.async_load_tmem)

By The JAX authors

© Copyright 2024, The JAX Authors.\
