- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.ones_like

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.ones_like.rst "Download source file")
-  .pdf

# jax.numpy.ones_like

## Contents

- [`ones_like()`](#jax.numpy.ones_like)

# jax.numpy.ones_like[\#](#jax-numpy-ones-like "Link to this heading")

jax.numpy.ones_like(*a*, *dtype=None*, *shape=None*, *\**, *device=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/array_creation.py#L313-L361)[\#](#jax.numpy.ones_like "Link to this definition")  
Create an array of ones with the same shape and dtype as an array.

JAX implementation of [`numpy.ones_like()`](https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html#numpy.ones_like "(in NumPy v2.4)").

Parameters:  
- **a** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)") *\|* *DuckTypedArray*) – Array-like object with `shape` and `dtype` attributes.

- **shape** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")) – optionally override the shape of the created array.

- **dtype** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* [*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") *\|* *SupportsDType* *\|* *None*) – optionally override the dtype of the created array.

- **device** ([*Device*](jax.Device.html#jax.Device "jaxlib._jax.Device") *\|* [*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jaxlib._jax.Sharding") *\|* *None*) – (optional) [`Device`](jax.Device.html#jax.Device "jax.Device") or [`Sharding`](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") to which the created array will be committed.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*)

Returns:  
Array of the specified shape and dtype, on the specified device if specified.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.empty()`](jax.numpy.empty.html#jax.numpy.empty "jax.numpy.empty")

- [`jax.numpy.zeros_like()`](jax.numpy.zeros_like.html#jax.numpy.zeros_like "jax.numpy.zeros_like")

- [`jax.numpy.ones_like()`](#jax.numpy.ones_like "jax.numpy.ones_like")

- [`jax.numpy.full_like()`](jax.numpy.full_like.html#jax.numpy.full_like "jax.numpy.full_like")

Examples

    >>> x = jnp.arange(4)
    >>> jnp.ones_like(x)
    Array([1, 1, 1, 1], dtype=int32)
    >>> jnp.ones_like(x, dtype=bool)
    Array([ True,  True,  True,  True], dtype=bool)
    >>> jnp.ones_like(x, shape=(2, 3))
    Array([[1, 1, 1],
           [1, 1, 1]], dtype=int32)

[](jax.numpy.ones.html "previous page")

previous

jax.numpy.ones

[](jax.numpy.outer.html "next page")

next

jax.numpy.outer

Contents

- [`ones_like()`](#jax.numpy.ones_like)

By The JAX authors

© Copyright 2024, The JAX Authors.\
