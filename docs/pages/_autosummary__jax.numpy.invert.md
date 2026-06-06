- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.invert

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.invert.rst "Download source file")
-  .pdf

# jax.numpy.invert

## Contents

- [`invert()`](#jax.numpy.invert)

# jax.numpy.invert[\#](#jax-numpy-invert "Link to this heading")

jax.numpy.invert(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L143-L189)[\#](#jax.numpy.invert "Link to this definition")  
Compute the bitwise inversion of an input.

JAX implementation of `numpy.invert()`. This function provides the implementation of the `~` operator for JAX arrays.

Parameters:  
**x** (*ArrayLike*) – input array, must be boolean or integer typed.

Returns:  
An array of the same shape and dtype as `` `x ``, with the bits inverted.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.bitwise_invert()`](jax.numpy.bitwise_invert.html#jax.numpy.bitwise_invert "jax.numpy.bitwise_invert"): Array API alias of this function.

- [`jax.numpy.logical_not()`](jax.numpy.logical_not.html#jax.numpy.logical_not "jax.numpy.logical_not"): Invert after casting input to boolean.

Examples

    >>> x = jnp.arange(5, dtype='uint8')
    >>> print(x)
    [0 1 2 3 4]
    >>> print(jnp.invert(x))
    [255 254 253 252 251]

This function implements the unary `~` operator for JAX arrays:

    >>> print(~x)
    [255 254 253 252 251]

[`invert()`](#jax.numpy.invert "jax.numpy.invert") operates bitwise on the input, and so the meaning of its output may be more clear by showing the bitwise representation:

    >>> with jnp.printoptions(formatter={'int': lambda x: format(x, '#010b')}):
    ...   print(f"{x  = }")
    ...   print(f"{~x = }")
    x  = Array([0b00000000, 0b00000001, 0b00000010, 0b00000011, 0b00000100], dtype=uint8)
    ~x = Array([0b11111111, 0b11111110, 0b11111101, 0b11111100, 0b11111011], dtype=uint8)

For boolean inputs, [`invert()`](#jax.numpy.invert "jax.numpy.invert") is equivalent to [`logical_not()`](jax.numpy.logical_not.html#jax.numpy.logical_not "jax.numpy.logical_not"):

    >>> x = jnp.array([True, False, True, True, False])
    >>> jnp.invert(x)
    Array([False,  True, False, False,  True], dtype=bool)

[](jax.numpy.intersect1d.html "previous page")

previous

jax.numpy.intersect1d

[](jax.numpy.isclose.html "next page")

next

jax.numpy.isclose

Contents

- [`invert()`](#jax.numpy.invert)

By The JAX authors

© Copyright 2024, The JAX Authors.\
