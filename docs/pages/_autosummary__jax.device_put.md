- [](../index.html)
- [API Reference](../jax.html)
- jax.device_put

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.device_put.rst "Download source file")
-  .pdf

# jax.device_put

## Contents

- [`device_put()`](#jax.device_put)

# jax.device_put[\#](#jax-device-put "Link to this heading")

jax.device_put(*x*, *device=None*, *\**, *src=None*, *donate=False*, *may_alias=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L2028-L2119)[\#](#jax.device_put "Link to this definition")  
Transfers `x` to `device`.

Parameters:  
- **x** – An array, scalar, or (nested) standard Python container thereof.

- **device** (*None* *\|* *xc.Device* *\|* [*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") *\|* *P* *\|* *Format* *\|* *Any*) – The (optional) [`Device`](jax.Device.html#jax.Device "jax.Device"), `Sharding`, or a (nested) `Sharding` in standard Python container (must be a tree prefix of `x`), representing the device(s) to which `x` should be transferred. If given, then the result is committed to the device(s).

- **src** (*None* *\|* *xc.Device* *\|* [*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") *\|* *P* *\|* *Format* *\|* *Any*) – The (optional) [`Device`](jax.Device.html#jax.Device "jax.Device"), `Sharding`, or a (nested) `Sharding` in standard Python container (must be a tree prefix of `x`), representing the device(s) on which `x` belongs.

- **donate** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *Any*) – bool or a (nested) bool in standard Python container (must be a tree prefix of `x`). If True, `x` can be overwritten and marked deleted in the caller. This is best effort. JAX will donate if possible, otherwise it won’t. The input buffer (in the future) will always be deleted if donated.

- **may_alias** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None* *\|* *Any*) – bool or None or a (nested) bool in standard Python container (must be a tree prefix of `x`). If False, x will be copied. If true, x may be aliased depending on the runtime’s implementation.

Returns:  
A copy of `x` that resides on `device`.

If the `device` parameter is `None`, then this operation behaves like the identity function if the operand is on any device already, otherwise it transfers the data to the default device, uncommitted.

This function is always asynchronous, i.e. returns immediately without blocking the calling Python thread until any transfers are completed.

[](jax.ShapeDtypeStruct.html "previous page")

previous

jax.ShapeDtypeStruct

[](jax.device_get.html "next page")

next

jax.device_get

Contents

- [`device_put()`](#jax.device_put)

By The JAX authors

© Copyright 2024, The JAX Authors.\
