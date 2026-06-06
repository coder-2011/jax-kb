- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.asarray

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.asarray.rst "Download source file")
-  .pdf

# jax.numpy.asarray

## Contents

- [`asarray()`](#jax.numpy.asarray)

# jax.numpy.asarray[\#](#jax-numpy-asarray "Link to this heading")

jax.numpy.asarray(*a*, *dtype=None*, *order=None*, *\**, *copy=None*, *device=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/array_constructors.py#L374-L455)[\#](#jax.numpy.asarray "Link to this definition")  
Convert an object to a JAX array.

JAX implementation of [`numpy.asarray()`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html#numpy.asarray "(in NumPy v2.4)").

Parameters:  
- **a** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")) – an object that is convertible to an array. This includes JAX arrays, NumPy arrays, Python scalars, Python collections like lists and tuples, objects with a `__jax_array__` method, and objects supporting the Python buffer protocol.

- **dtype** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* [*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") *\|* *SupportsDType* *\|* *None*) – optionally specify the dtype of the output array. If not specified it will be inferred from the input.

- **order** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – not implemented in JAX

- **copy** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None*) – optional boolean specifying the copy mode. If True, then always return a copy. If False, then error if a copy is necessary. Default is None, which will only copy when necessary.

- **device** ([*Device*](jax.Device.html#jax.Device "jaxlib._jax.Device") *\|* [*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jaxlib._jax.Sharding") *\|* *None*) – optional [`Device`](jax.Device.html#jax.Device "jax.Device") or [`Sharding`](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") to which the created array will be committed.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*)

Returns:  
A JAX array constructed from the input.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.array()`](jax.numpy.array.html#jax.numpy.array "jax.numpy.array"): like asarray, but defaults to copy=True.

- [`jax.numpy.from_dlpack()`](jax.numpy.from_dlpack.html#jax.numpy.from_dlpack "jax.numpy.from_dlpack"): construct a JAX array from an object that implements the dlpack interface.

- [`jax.numpy.frombuffer()`](jax.numpy.frombuffer.html#jax.numpy.frombuffer "jax.numpy.frombuffer"): construct a JAX array from an object that implements the buffer interface.

Examples

Constructing JAX arrays from Python scalars:

    >>> jnp.asarray(True)
    Array(True, dtype=bool)
    >>> jnp.asarray(42)
    Array(42, dtype=int32, weak_type=True)
    >>> jnp.asarray(3.5)
    Array(3.5, dtype=float32, weak_type=True)
    >>> jnp.asarray(1 + 1j)
    Array(1.+1.j, dtype=complex64, weak_type=True)

Constructing JAX arrays from Python collections:

    >>> jnp.asarray([1, 2, 3])  # list of ints -> 1D array
    Array([1, 2, 3], dtype=int32)
    >>> jnp.asarray([(1, 2, 3), (4, 5, 6)])  # list of tuples of ints -> 2D array
    Array([[1, 2, 3],
           [4, 5, 6]], dtype=int32)
    >>> jnp.asarray(range(5))
    Array([0, 1, 2, 3, 4], dtype=int32)

Constructing JAX arrays from NumPy arrays:

    >>> jnp.asarray(np.linspace(0, 2, 5))
    Array([0. , 0.5, 1. , 1.5, 2. ], dtype=float32)

Constructing a JAX array via the Python buffer interface, using Python’s built-in [`array`](https://docs.python.org/3/library/array.html#module-array "(in Python v3.14)") module.

    >>> from array import array
    >>> pybuffer = array('i', [2, 3, 5, 7])
    >>> jnp.asarray(pybuffer)
    Array([2, 3, 5, 7], dtype=int32)

[](jax.numpy.array_str.html "previous page")

previous

jax.numpy.array_str

[](jax.numpy.asin.html "next page")

next

jax.numpy.asin

Contents

- [`asarray()`](#jax.numpy.asarray)

By The JAX authors

© Copyright 2024, The JAX Authors.\
