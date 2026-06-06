- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.full

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.full.rst "Download source file")
-  .pdf

# jax.numpy.full

## Contents

- [`full()`](#jax.numpy.full)

# jax.numpy.full[\#](#jax-numpy-full "Link to this heading")

jax.numpy.full(*shape*, *fill_value*, *dtype=None*, *\**, *device=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/array_creation.py#L206-L261)[\#](#jax.numpy.full "Link to this definition")  
Create an array full of a specified value.

JAX implementation of [`numpy.full()`](https://numpy.org/doc/stable/reference/generated/numpy.full.html#numpy.full "(in NumPy v2.4)").

Parameters:  
- **shape** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")) – int or sequence of ints specifying the shape of the created array.

- **fill_value** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – scalar or array with which to fill the created array.

- **dtype** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* [*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") *\|* *SupportsDType* *\|* *None*) – optional dtype for the created array; defaults to the dtype of the fill value.

- **device** ([*Device*](jax.Device.html#jax.Device "jaxlib._jax.Device") *\|* [*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jaxlib._jax.Sharding") *\|* *None*) – (optional) [`Device`](jax.Device.html#jax.Device "jax.Device") or [`Sharding`](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") to which the created array will be committed.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*)

Returns:  
Array of the specified shape and dtype, on the specified device if specified.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.full_like()`](jax.numpy.full_like.html#jax.numpy.full_like "jax.numpy.full_like")

- [`jax.numpy.empty()`](jax.numpy.empty.html#jax.numpy.empty "jax.numpy.empty")

- [`jax.numpy.zeros()`](jax.numpy.zeros.html#jax.numpy.zeros "jax.numpy.zeros")

- [`jax.numpy.ones()`](jax.numpy.ones.html#jax.numpy.ones "jax.numpy.ones")

Examples

    >>> jnp.full(4, 2, dtype=float)
    Array([2., 2., 2., 2.], dtype=float32)
    >>> jnp.full((2, 3), 0, dtype=bool)
    Array([[False, False, False],
           [False, False, False]], dtype=bool)

fill_value may also be an array that is broadcast to the specified shape:

    >>> jnp.full((2, 3), fill_value=jnp.arange(3))
    Array([[0, 1, 2],
           [0, 1, 2]], dtype=int32)

[](jax.numpy.from_dlpack.html "previous page")

previous

jax.numpy.from_dlpack

[](jax.numpy.full_like.html "next page")

next

jax.numpy.full_like

Contents

- [`full()`](#jax.numpy.full)

By The JAX authors

© Copyright 2024, The JAX Authors.\
