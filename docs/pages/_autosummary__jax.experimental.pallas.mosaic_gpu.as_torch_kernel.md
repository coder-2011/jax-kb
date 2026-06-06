- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.as_torch_kernel

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.as_torch_kernel.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.as_torch_kernel

## Contents

- [`as_torch_kernel()`](#jax.experimental.pallas.mosaic_gpu.as_torch_kernel)

# jax.experimental.pallas.mosaic_gpu.as_torch_kernel[\#](#jax-experimental-pallas-mosaic-gpu-as-torch-kernel "Link to this heading")

jax.experimental.pallas.mosaic_gpu.as_torch_kernel(*fn*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/torch.py#L38-L77)[\#](#jax.experimental.pallas.mosaic_gpu.as_torch_kernel "Link to this definition")  
Makes a Mosaic GPU kernel callable with PyTorch tensors.

Parameters:  
**fn** – A JAX function that invokes a Mosaic GPU kernel. Note that the implementation currently only supports functions that contain a single Mosaic GPU kernel invocation, without any other JAX API calls, e.g. from [`jax.numpy`](../jax.numpy.html#module-jax.numpy "jax.numpy").

Returns:  
A wrapper function that accepts PyTorch tensors as inputs and returns PyTorch tensors as outputs. The output tensors are allocated on the same device as the input tensors.

Example:

    @functools.partial(
        plgpu.kernel, out_shape=jax.ShapeDtypeStruct([128], jnp.int32)
    )
    def kernel(x_gmem_ref, y_gmem_ref, o_gmem_ref):
      ...

    x = torch.arange(128, dtype=torch.int32, device="cuda")
    y = x * x
    out = plgpu.as_torch_kernel(kernel)(x, y)

[](jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef

[](jax.experimental.pallas.mosaic_gpu.kernel.html "next page")

next

jax.experimental.pallas.mosaic_gpu.kernel

Contents

- [`as_torch_kernel()`](#jax.experimental.pallas.mosaic_gpu.as_torch_kernel)

By The JAX authors

© Copyright 2024, The JAX Authors.\
