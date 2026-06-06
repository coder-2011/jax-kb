- [](../index.html)
- [API Reference](../jax.html)
- jax.Array.copy_to_host_async

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.Array.copy_to_host_async.rst "Download source file")
-  .pdf

# jax.Array.copy_to_host_async

## Contents

- [`Array.copy_to_host_async()`](#jax.Array.copy_to_host_async)

# jax.Array.copy_to_host_async[\#](#jax-array-copy-to-host-async "Link to this heading")

Array.copy_to_host_async()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/basearray.py#L170-L185)[\#](#jax.Array.copy_to_host_async "Link to this definition")  
Copies an `Array` to the host asynchronously.

For arrays that live an an accelerator, such as a GPU or a TPU, JAX may cache the value of the array on the host. Normally this happens behind the scenes when the value of an on-device array is requested by the user, but waiting to initiate a device-to-host copy until the value is requested requires that JAX block the caller while waiting for the copy to complete.

`copy_to_host_async` requests that JAX populate its on-host cache of an array, but does not wait for the copy to complete. This may speed up a future on-host access to the array’s contents.

[](jax.Array.copy.html "previous page")

previous

jax.Array.copy

[](jax.Array.cumprod.html "next page")

next

jax.Array.cumprod

Contents

- [`Array.copy_to_host_async()`](#jax.Array.copy_to_host_async)

By The JAX authors

© Copyright 2024, The JAX Authors.\
