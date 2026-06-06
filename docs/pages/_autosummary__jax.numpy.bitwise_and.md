- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.bitwise_and

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.bitwise_and.rst "Download source file")
-  .pdf

# jax.numpy.bitwise_and

## Contents

- [`bitwise_and`](#jax.numpy.bitwise_and)

# jax.numpy.bitwise_and[\#](#jax-numpy-bitwise-and "Link to this heading")

jax.numpy.bitwise_and *= \<jnp.ufunc 'bitwise_and'\>*[\#](#jax.numpy.bitwise_and "Link to this definition")  
Compute the bitwise AND operation elementwise.

JAX implementation of [`numpy.bitwise_and`](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_and.html#numpy.bitwise_and "(in NumPy v2.4)"). This is a universal function, and supports the additional APIs described at [`jax.numpy.ufunc`](jax.numpy.ufunc.html#jax.numpy.ufunc "jax.numpy.ufunc"). This function provides the implementation of the `&` operator for JAX arrays.

Parameters:  
- **x** – integer or boolean arrays. Must be broadcastable to a common shape.

- **y** – integer or boolean arrays. Must be broadcastable to a common shape.

- **args** (*ArrayLike*)

- **out** (*None*)

- **where** (*None*)

Returns:  
Array containing the result of the element-wise bitwise AND.

Return type:  
Any

Examples

Calling `bitwise_and` explicitly:

    >>> x = jnp.arange(4)
    >>> jnp.bitwise_and(x, 1)
    Array([0, 1, 0, 1], dtype=int32)

Calling `bitwise_and` via the `&` operator:

    >>> x & 1
    Array([0, 1, 0, 1], dtype=int32)

[](jax.numpy.bincount.html "previous page")

previous

jax.numpy.bincount

[](jax.numpy.bitwise_count.html "next page")

next

jax.numpy.bitwise_count

Contents

- [`bitwise_and`](#jax.numpy.bitwise_and)

By The JAX authors

© Copyright 2024, The JAX Authors.\
