- [](../index.html)
- [API Reference](../jax.html)
- jax.Array.byteswap

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.Array.byteswap.rst "Download source file")
-  .pdf

# jax.Array.byteswap

## Contents

- [`Array.byteswap()`](#jax.Array.byteswap)

# jax.Array.byteswap[\#](#jax-array-byteswap "Link to this heading")

*abstract* Array.byteswap()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/array_methods.py#L127-L164)[\#](#jax.Array.byteswap "Link to this definition")  
Swap the bytes of the array elements.

This switches between a little-endian and big-endian data representation.

Returns:  
An array with the same dtype as `self`, with underlying bytes of each entry reversed.

Parameters:  
**self** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> import jax.numpy as jnp
    >>> x = jnp.arange(5, dtype='int32')
    >>> x
    Array([0, 1, 2, 3, 4], dtype=int32)
    >>> x.byteswap()
    Array([       0, 16777216, 33554432, 50331648, 67108864], dtype=int32)

When the resulting bytes are viewed as a big-endian dtype (possible in NumPy, but not in JAX) they represent the original values:

    >>> import numpy as np
    >>> np.array(x.byteswap()).view('>i4')  # view as big-endian
    array([0, 1, 2, 3, 4], dtype='>i4')

Calling byteswap twice will return the original array:

    >>> x.byteswap().byteswap()
    Array([0, 1, 2, 3, 4], dtype=int32)

[](jax.Array.at.html "previous page")

previous

jax.Array.at

[](jax.Array.choose.html "next page")

next

jax.Array.choose

Contents

- [`Array.byteswap()`](#jax.Array.byteswap)

By The JAX authors

© Copyright 2024, The JAX Authors.\
