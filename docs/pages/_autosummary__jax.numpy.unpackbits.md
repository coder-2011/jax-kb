- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.unpackbits

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.unpackbits.rst "Download source file")
-  .pdf

# jax.numpy.unpackbits

## Contents

- [`unpackbits()`](#jax.numpy.unpackbits)

# jax.numpy.unpackbits[\#](#jax-numpy-unpackbits "Link to this heading")

jax.numpy.unpackbits(*a*, *axis=None*, *count=None*, *bitorder='big'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L8700-L8789)[\#](#jax.numpy.unpackbits "Link to this definition")  
Unpack the bits in a uint8 array.

JAX implementation of [`numpy.unpackbits()`](https://numpy.org/doc/stable/reference/generated/numpy.unpackbits.html#numpy.unpackbits "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – N-dimensional array of type `uint8`.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional axis along which to unpack. If not specified, `a` will be flattened

- **count** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – specify the number of bits to unpack (if positive) or the number of bits to trim from the end (if negative).

- **bitorder** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – `"big"` (default) or `"little"`: specify whether the bit order is big-endian or little-endian.

Returns:  
a uint8 array of unpacked bits.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.packbits()`](jax.numpy.packbits.html#jax.numpy.packbits "jax.numpy.packbits"): this inverse of `unpackbits`.

Examples

Unpacking bits from a scalar:

    >>> jnp.unpackbits(jnp.uint8(27))  # big-endian by default
    Array([0, 0, 0, 1, 1, 0, 1, 1], dtype=uint8)
    >>> jnp.unpackbits(jnp.uint8(27), bitorder="little")
    Array([1, 1, 0, 1, 1, 0, 0, 0], dtype=uint8)

Compare this to the Python binary representation:

    >>> 0b00011011
    27

Unpacking bits along an axis:

    >>> vals = jnp.array([[154],
    ...                   [ 49]], dtype='uint8')
    >>> bits = jnp.unpackbits(vals, axis=1)
    >>> bits
    Array([[1, 0, 0, 1, 1, 0, 1, 0],
           [0, 0, 1, 1, 0, 0, 0, 1]], dtype=uint8)

Using [`packbits()`](jax.numpy.packbits.html#jax.numpy.packbits "jax.numpy.packbits") to invert this:

    >>> jnp.packbits(bits, axis=1)
    Array([[154],
           [ 49]], dtype=uint8)

The `count` keyword lets `unpackbits` serve as an inverse of `packbits` in cases where not all bits are present:

    >>> bits = jnp.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1])  # 11 bits
    >>> vals = jnp.packbits(bits)
    >>> vals
    Array([219,  96], dtype=uint8)
    >>> jnp.unpackbits(vals)  # 16 zero-padded bits
    Array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0], dtype=uint8)
    >>> jnp.unpackbits(vals, count=11)  # specify 11 output bits
    Array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=uint8)
    >>> jnp.unpackbits(vals, count=-5)  # specify 5 bits to be trimmed
    Array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=uint8)

[](jax.numpy.unique_values.html "previous page")

previous

jax.numpy.unique_values

[](jax.numpy.unravel_index.html "next page")

next

jax.numpy.unravel_index

Contents

- [`unpackbits()`](#jax.numpy.unpackbits)

By The JAX authors

© Copyright 2024, The JAX Authors.\
