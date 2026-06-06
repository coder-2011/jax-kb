- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.from_dlpack

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.from_dlpack.rst "Download source file")
-  .pdf

# jax.numpy.from_dlpack

## Contents

- [`from_dlpack()`](#jax.numpy.from_dlpack)

# jax.numpy.from_dlpack[\#](#jax-numpy-from-dlpack "Link to this heading")

jax.numpy.from_dlpack(*x*, */*, *\**, *device=None*, *copy=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L5540-L5599)[\#](#jax.numpy.from_dlpack "Link to this definition")  
Construct a JAX array via DLPack.

JAX implementation of [`numpy.from_dlpack()`](https://numpy.org/doc/stable/reference/generated/numpy.from_dlpack.html#numpy.from_dlpack "(in NumPy v2.4)").

Parameters:  
- **x** (*Any*) – An object that implements the [DLPack](https://dmlc.github.io/dlpack) protocol via the `__dlpack__` and `__dlpack_device__` methods, or a legacy DLPack tensor on either CPU or GPU.

- **device** (*xc.Device* *\|* [*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") *\|* *None*) – An optional [`Device`](jax.Device.html#jax.Device "jax.Device") or [`Sharding`](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding"), representing the single device onto which the returned array should be placed. If given, then the result is committed to the device. If unspecified, the resulting array will be unpacked onto the same device it originated from. Setting `device` to a device different from the source of `external_array` will require a copy, meaning `copy` must be set to either `True` or `None`.

- **copy** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None*) – An optional boolean, controlling whether or not a copy is performed. If `copy=True` then a copy is always performed, even if unpacked onto the same device. If `copy=False` then the copy is never performed and will raise an error if necessary. When `copy=None` (default) then a copy may be performed if needed for a device transfer.

Returns:  
A JAX array of the input buffer.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

While JAX arrays are always immutable, dlpack buffers cannot be marked as immutable, and it is possible for processes external to JAX to mutate them in-place. If a JAX Array is constructed from a dlpack buffer without copying and the source buffer is later modified in-place, it may lead to undefined behavior when using the associated JAX array.

Examples

Passing data between NumPy and JAX via [DLPack](https://dmlc.github.io/dlpack):

    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> x_numpy = rng.random(4, dtype='float32')
    >>> print(x_numpy)
    [0.08925092 0.773956   0.6545715  0.43887842]
    >>> hasattr(x_numpy, "__dlpack__")  # NumPy supports the DLPack interface
    True

    >>> import jax.numpy as jnp
    >>> x_jax = jnp.from_dlpack(x_numpy)
    >>> print(x_jax)
    [0.08925092 0.773956   0.6545715  0.43887842]
    >>> hasattr(x_jax, "__dlpack__")  # JAX supports the DLPack interface
    True

    >>> x_numpy_round_trip = np.from_dlpack(x_jax)
    >>> print(x_numpy_round_trip)
    [0.08925092 0.773956   0.6545715  0.43887842]

[](jax.numpy.fromstring.html "previous page")

previous

jax.numpy.fromstring

[](jax.numpy.full.html "next page")

next

jax.numpy.full

Contents

- [`from_dlpack()`](#jax.numpy.from_dlpack)

By The JAX authors

© Copyright 2024, The JAX Authors.\
