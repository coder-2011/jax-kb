- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.bitwise_xor

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.bitwise_xor.rst "Download source file")
-  .pdf

# jax.numpy.bitwise_xor

## Contents

- [`bitwise_xor`](#jax.numpy.bitwise_xor)

# jax.numpy.bitwise_xor[\#](#jax-numpy-bitwise-xor "Link to this heading")

jax.numpy.bitwise_xor *= \<jnp.ufunc 'bitwise_xor'\>*[\#](#jax.numpy.bitwise_xor "Link to this definition")  
Compute the bitwise XOR operation elementwise.

JAX implementation of [`numpy.bitwise_xor`](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_xor.html#numpy.bitwise_xor "(in NumPy v2.4)"). This is a universal function, and supports the additional APIs described at [`jax.numpy.ufunc`](jax.numpy.ufunc.html#jax.numpy.ufunc "jax.numpy.ufunc"). This function provides the implementation of the `^` operator for JAX arrays.

Parameters:  
- **x** – integer or boolean arrays. Must be broadcastable to a common shape.

- **y** – integer or boolean arrays. Must be broadcastable to a common shape.

- **args** (*ArrayLike*)

- **out** (*None*)

- **where** (*None*)

Returns:  
Array containing the result of the element-wise bitwise XOR.

Return type:  
Any

Examples

Calling `bitwise_xor` explicitly:

    >>> x = jnp.arange(4)
    >>> jnp.bitwise_xor(x, 1)
    Array([1, 0, 3, 2], dtype=int32)

Calling `bitwise_xor` via the `^` operator:

    >>> x ^ 1
    Array([1, 0, 3, 2], dtype=int32)

[](jax.numpy.bitwise_right_shift.html "previous page")

previous

jax.numpy.bitwise_right_shift

[](jax.numpy.blackman.html "next page")

next

jax.numpy.blackman

Contents

- [`bitwise_xor`](#jax.numpy.bitwise_xor)

By The JAX authors

© Copyright 2024, The JAX Authors.\
