- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.empty

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.empty.rst "Download source file")
-  .pdf

# jax.numpy.empty

## Contents

- [`empty()`](#jax.numpy.empty)

# jax.numpy.empty[\#](#jax-numpy-empty "Link to this heading")

jax.numpy.empty(*shape*, *dtype=None*, *\**, *device=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/array_creation.py#L147-L196)[\#](#jax.numpy.empty "Link to this definition")  
Create an empty array.

JAX implementation of [`numpy.empty()`](https://numpy.org/doc/stable/reference/generated/numpy.empty.html#numpy.empty "(in NumPy v2.4)").

Note

For historical reasons, [`jax.numpy.empty()`](#jax.numpy.empty "jax.numpy.empty") is currently equivalent to [`jax.numpy.zeros()`](jax.numpy.zeros.html#jax.numpy.zeros "jax.numpy.zeros"): i.e. it returns a buffer initialized with zeros. To create a buffer of uninitialized values, please use [`jax.lax.empty()`](jax.lax.empty.html#jax.lax.empty "jax.lax.empty").

Parameters:  
- **shape** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")) – int or sequence of ints specifying the shape of the created array.

- **dtype** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* [*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") *\|* *SupportsDType* *\|* *None*) – optional dtype for the created array; defaults to float32 or float64 depending on the X64 configuration (see [Default dtypes and the X64 flag](../default_dtypes.html#default-dtypes)).

- **device** ([*Device*](jax.Device.html#jax.Device "jaxlib._jax.Device") *\|* [*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jaxlib._jax.Sharding") *\|* *None*) – (optional) [`Device`](jax.Device.html#jax.Device "jax.Device") or [`Sharding`](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") to which the created array will be committed. This argument exists for compatibility with the [Python Array API standard](../jax.numpy.html#python-array-api).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – (optional) [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") or `NamedSharding` representing the sharding of the created array (see [explicit sharding](https://docs.jax.dev/en/latest/parallel.html) for more details). This argument exists for consistency with other array creation routines across JAX. Specifying both `out_sharding` and `device` will result in an error.

Returns:  
Array of the specified shape and dtype, with the given device/sharding if specified.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.empty()`](jax.lax.empty.html#jax.lax.empty "jax.lax.empty")

- [`jax.numpy.empty_like()`](jax.numpy.empty_like.html#jax.numpy.empty_like "jax.numpy.empty_like")

- [`jax.numpy.zeros()`](jax.numpy.zeros.html#jax.numpy.zeros "jax.numpy.zeros")

- [`jax.numpy.ones()`](jax.numpy.ones.html#jax.numpy.ones "jax.numpy.ones")

- [`jax.numpy.full()`](jax.numpy.full.html#jax.numpy.full "jax.numpy.full")

Examples

    >>> jnp.empty(4)  
    Array([0., 0., 0., 0.], dtype=float32)
    >>> jnp.empty((2, 3), dtype=bool)  
    Array([[False, False, False],
           [False, False, False]], dtype=bool)

[](jax.numpy.einsum_path.html "previous page")

previous

jax.numpy.einsum_path

[](jax.numpy.empty_like.html "next page")

next

jax.numpy.empty_like

Contents

- [`empty()`](#jax.numpy.empty)

By The JAX authors

© Copyright 2024, The JAX Authors.\
