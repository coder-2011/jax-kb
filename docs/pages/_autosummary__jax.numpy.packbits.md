- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.packbits

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.packbits.rst "Download source file")
-  .pdf

# jax.numpy.packbits

## Contents

- [`packbits()`](#jax.numpy.packbits)

# jax.numpy.packbits[\#](#jax-numpy-packbits "Link to this heading")

jax.numpy.packbits(*a*, *axis=None*, *bitorder='big'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L8616-L8698)[\#](#jax.numpy.packbits "Link to this definition")  
Pack array of bits into a uint8 array.

JAX implementation of [`numpy.packbits()`](https://numpy.org/doc/stable/reference/generated/numpy.packbits.html#numpy.packbits "(in NumPy v2.4)")

Parameters:  
- **a** (*ArrayLike*) – N-dimensional array of bits to pack.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional axis along which to pack bits. If not specified, `a` will be flattened.

- **bitorder** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – `"big"` (default) or `"little"`: specify whether the bit order is big-endian or little-endian.

Returns:  
A uint8 array of packed values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.unpackbits()`](jax.numpy.unpackbits.html#jax.numpy.unpackbits "jax.numpy.unpackbits"): inverse of `packbits`.

Examples

Packing bits in one dimension:

    >>> bits = jnp.array([0, 0, 0, 0, 0, 1, 1, 1])
    >>> jnp.packbits(bits)
    Array([7], dtype=uint8)
    >>> 0b00000111  # equivalent bit-wise representation:
    7

Optionally specifying little-endian convention:

    >>> jnp.packbits(bits, bitorder="little")
    Array([224], dtype=uint8)
    >>> 0b11100000  # equivalent bit-wise representation
    224

If the number of bits is not a multiple of 8, it will be right-padded with zeros:

    >>> jnp.packbits(jnp.array([1, 0, 1]))
    Array([160], dtype=uint8)
    >>> jnp.packbits(jnp.array([1, 0, 1, 0, 0, 0, 0, 0]))
    Array([160], dtype=uint8)

For a multi-dimensional input, bits may be packed along a specified axis:

    >>> a = jnp.array([[1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
    ...                [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1]])
    >>> vals = jnp.packbits(a, axis=1)
    >>> vals
    Array([[212, 150],
           [ 69, 207]], dtype=uint8)

The inverse of `packbits` is provided by [`unpackbits()`](jax.numpy.unpackbits.html#jax.numpy.unpackbits "jax.numpy.unpackbits"):

    >>> jnp.unpackbits(vals, axis=1)
    Array([[1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
           [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1]], dtype=uint8)

[](jax.numpy.outer.html "previous page")

previous

jax.numpy.outer

[](jax.numpy.pad.html "next page")

next

jax.numpy.pad

Contents

- [`packbits()`](#jax.numpy.packbits)

By The JAX authors

© Copyright 2024, The JAX Authors.\
