- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.frombuffer

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.frombuffer.rst "Download source file")
-  .pdf

# jax.numpy.frombuffer

## Contents

- [`frombuffer()`](#jax.numpy.frombuffer)

# jax.numpy.frombuffer[\#](#jax-numpy-frombuffer "Link to this heading")

jax.numpy.frombuffer(*buffer*, *dtype=\<class 'float'\>*, *count=-1*, *offset=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L5454-L5500)[\#](#jax.numpy.frombuffer "Link to this definition")  
Convert a buffer into a 1-D JAX array.

JAX implementation of [`numpy.frombuffer()`](https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html#numpy.frombuffer "(in NumPy v2.4)").

Parameters:  
- **buffer** ([*bytes*](https://docs.python.org/3/library/stdtypes.html#bytes "(in Python v3.14)") *\|* *Any*) – an object containing the data. It must be either a bytes object with a length that is an integer multiple of the dtype element size, or it must be an object exporting the [Python buffer interface](https://docs.python.org/3/c-api/buffer.html).

- **dtype** (*DTypeLike*) – optional. Desired data type for the array. Default is `float64`. This specifies the dtype used to parse the buffer, but note that after parsing, 64-bit values will be cast to 32-bit JAX arrays if the `jax_enable_x64` flag is set to `False`.

- **count** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – optional integer specifying the number of items to read from the buffer. If -1 (default), all items from the buffer are read.

- **offset** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – optional integer specifying the number of bytes to skip at the beginning of the buffer. Default is 0.

Returns:  
A 1-D JAX array representing the interpreted data from the buffer.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.fromstring()`](jax.numpy.fromstring.html#jax.numpy.fromstring "jax.numpy.fromstring"): convert a string of text into 1-D JAX array.

Examples

Using a bytes buffer:

    >>> buf = b"\x00\x01\x02\x03\x04"
    >>> jnp.frombuffer(buf, dtype=jnp.uint8)
    Array([0, 1, 2, 3, 4], dtype=uint8)
    >>> jnp.frombuffer(buf, dtype=jnp.uint8, offset=1)
    Array([1, 2, 3, 4], dtype=uint8)

Constructing a JAX array via the Python buffer interface, using Python’s built-in [`array`](https://docs.python.org/3/library/array.html#module-array "(in Python v3.14)") module.

    >>> from array import array
    >>> pybuffer = array('i', [0, 1, 2, 3, 4])
    >>> jnp.frombuffer(pybuffer, dtype=jnp.int32)
    Array([0, 1, 2, 3, 4], dtype=int32)

[](jax.numpy.frexp.html "previous page")

previous

jax.numpy.frexp

[](jax.numpy.fromfile.html "next page")

next

jax.numpy.fromfile

Contents

- [`frombuffer()`](#jax.numpy.frombuffer)

By The JAX authors

© Copyright 2024, The JAX Authors.\
