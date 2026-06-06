- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.bitwise_count

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.bitwise_count.rst "Download source file")
-  .pdf

# jax.numpy.bitwise_count

## Contents

- [`bitwise_count()`](#jax.numpy.bitwise_count)

# jax.numpy.bitwise_count[\#](#jax-numpy-bitwise-count "Link to this heading")

jax.numpy.bitwise_count(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L2218-L2253)[\#](#jax.numpy.bitwise_count "Link to this definition")  
Counts the number of 1 bits in the binary representation of the absolute value of each element of `x`.

JAX implementation of [`numpy.bitwise_count`](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_count.html#numpy.bitwise_count "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – Input array, only accepts integer subtypes

Returns:  
An array-like object containing the binary 1 bit counts of the absolute value of each element in `x`, with the same shape as `x` of dtype uint8.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> x1 = jnp.array([64, 32, 31, 20])
    >>> # 64 = 0b1000000, 32 = 0b100000, 31 = 0b11111, 20 = 0b10100
    >>> jnp.bitwise_count(x1)
    Array([1, 1, 5, 2], dtype=uint8)

    >>> x2 = jnp.array([-16, -7, 7])
    >>> # |-16| = 0b10000, |-7| = 0b111, 7 = 0b111
    >>> jnp.bitwise_count(x2)
    Array([1, 3, 3], dtype=uint8)

    >>> x3 = jnp.array([[2, -7],[-9, 7]])
    >>> # 2 = 0b10, |-7| = 0b111, |-9| = 0b1001, 7 = 0b111
    >>> jnp.bitwise_count(x3)
    Array([[1, 3],
           [2, 3]], dtype=uint8)

[](jax.numpy.bitwise_and.html "previous page")

previous

jax.numpy.bitwise_and

[](jax.numpy.bitwise_invert.html "next page")

next

jax.numpy.bitwise_invert

Contents

- [`bitwise_count()`](#jax.numpy.bitwise_count)

By The JAX authors

© Copyright 2024, The JAX Authors.\
