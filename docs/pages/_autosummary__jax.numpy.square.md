- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.square

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.square.rst "Download source file")
-  .pdf

# jax.numpy.square

## Contents

- [`square()`](#jax.numpy.square)

# jax.numpy.square[\#](#jax-numpy-square "Link to this heading")

jax.numpy.square(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3193-L3241)[\#](#jax.numpy.square "Link to this definition")  
Calculate element-wise square of the input array.

JAX implementation of [`numpy.square`](https://numpy.org/doc/stable/reference/generated/numpy.square.html#numpy.square "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
An array containing the square of the elements of `x`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

`jnp.square` is equivalent to computing `jnp.power(x,`` ``2)`.

See also

- [`jax.numpy.sqrt()`](jax.numpy.sqrt.html#jax.numpy.sqrt "jax.numpy.sqrt"): Calculates the element-wise non-negative square root of the input array.

- [`jax.numpy.power()`](jax.numpy.power.html#jax.numpy.power "jax.numpy.power"): Calculates the element-wise base `x1` exponential of `x2`.

- [`jax.lax.integer_pow()`](jax.lax.integer_pow.html#jax.lax.integer_pow "jax.lax.integer_pow"): Computes element-wise power \\x^y\\, where \\y\\ is a fixed integer.

- [`jax.numpy.float_power()`](jax.numpy.float_power.html#jax.numpy.float_power "jax.numpy.float_power"): Computes the first array raised to the power of second array, element-wise, by promoting to the inexact dtype.

Examples

    >>> x = jnp.array([3, -2, 5.3, 1])
    >>> jnp.square(x)
    Array([ 9.      ,  4.      , 28.090002,  1.      ], dtype=float32)
    >>> jnp.power(x, 2)
    Array([ 9.      ,  4.      , 28.090002,  1.      ], dtype=float32)

For integer inputs:

    >>> x1 = jnp.array([2, 4, 5, 6])
    >>> jnp.square(x1)
    Array([ 4, 16, 25, 36], dtype=int32)

For complex-valued inputs:

    >>> x2 = jnp.array([1-3j, -1j, 2])
    >>> jnp.square(x2)
    Array([-8.-6.j, -1.+0.j,  4.+0.j], dtype=complex64)

[](jax.numpy.sqrt.html "previous page")

previous

jax.numpy.sqrt

[](jax.numpy.squeeze.html "next page")

next

jax.numpy.squeeze

Contents

- [`square()`](#jax.numpy.square)

By The JAX authors

© Copyright 2024, The JAX Authors.\
