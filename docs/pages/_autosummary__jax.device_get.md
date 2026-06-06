- [](../index.html)
- [API Reference](../jax.html)
- jax.device_get

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.device_get.rst "Download source file")
-  .pdf

# jax.device_get

## Contents

- [`device_get()`](#jax.device_get)

# jax.device_get[\#](#jax-device-get "Link to this heading")

jax.device_get(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L2270-L2308)[\#](#jax.device_get "Link to this definition")  
Transfer `x` to host.

If `x` is a pytree, then the individual buffers are copied in parallel.

Parameters:  
**x** (*Any*) – An array, scalar, Array or (nested) standard Python container thereof representing the array to be transferred to host.

Returns:  
An array or (nested) Python container thereof representing the value of `x`.

Examples

Passing a Array:

    >>> import jax
    >>> x = jax.numpy.array([1., 2., 3.])
    >>> jax.device_get(x)
    array([1., 2., 3.], dtype=float32)

Passing a scalar (has no effect):

    >>> jax.device_get(1)
    1

See also

- device_put

- device_put_sharded

- device_put_replicated

[](jax.device_put.html "previous page")

previous

jax.device_put

[](jax.default_backend.html "next page")

next

jax.default_backend

Contents

- [`device_get()`](#jax.device_get)

By The JAX authors

© Copyright 2024, The JAX Authors.\
