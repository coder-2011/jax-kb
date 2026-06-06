- [](../index.html)
- [API Reference](../jax.html)
- [`jax.dlpack` module](../jax.dlpack.html)
- jax.dlpack.from_dlpack

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.dlpack.from_dlpack.rst "Download source file")
-  .pdf

# jax.dlpack.from_dlpack

## Contents

- [`from_dlpack()`](#jax.dlpack.from_dlpack)

# jax.dlpack.from_dlpack[\#](#jax-dlpack-from-dlpack "Link to this heading")

jax.dlpack.from_dlpack(*external_array*, *device=None*, *copy=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/dlpack.py#L197-L289)[\#](#jax.dlpack.from_dlpack "Link to this definition")  
Returns a [`Array`](jax.Array.html#jax.Array "jax.Array") representation of a DLPack tensor.

The returned [`Array`](jax.Array.html#jax.Array "jax.Array") shares memory with `external_array` if no device transfer or copy was requested.

Parameters:  
- **external_array** – An array object that has `__dlpack__` and `__dlpack_device__` methods.

- **device** ([*\_jax.Device*](jax.Device.html#jax.Device "jaxlib._jax.Device") *\|* [*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") *\|* *None*) – The (optional) `Device`, representing the device on which the returned array should be placed. If given, then the result is committed to the device. If unspecified, the resulting array will be unpacked onto the same device it originated from. Setting `device` to a device different from the source of `external_array` will require a copy, meaning `copy` must be set to either `True` or `None`.

- **copy** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None*) – An (optional) boolean, controlling whether or not a copy is performed. If `copy=True` then a copy is always performed, even if unpacked onto the same device. If `copy=False` then the copy is never performed and will raise an error if necessary. When `copy=None` then a copy may be performed if needed for a device transfer.

Returns:  
A jax.Array

Note

While JAX arrays are always immutable, dlpack buffers cannot be marked as immutable, and it is possible for processes external to JAX to mutate them in-place. If a jax Array is constructed from a dlpack buffer and the buffer is later modified in-place, it may lead to undefined behavior when using the associated JAX array.

[](../jax.dlpack.html "previous page")

previous

`jax.dlpack` module

[](jax.dlpack.is_supported_dtype.html "next page")

next

jax.dlpack.is_supported_dtype

Contents

- [`from_dlpack()`](#jax.dlpack.from_dlpack)

By The JAX authors

© Copyright 2024, The JAX Authors.\
