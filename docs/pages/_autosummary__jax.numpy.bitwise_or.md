- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.bitwise_or

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.bitwise_or.rst "Download source file")
-  .pdf

# jax.numpy.bitwise_or

## Contents

- [`bitwise_or`](#jax.numpy.bitwise_or)

# jax.numpy.bitwise_or[\#](#jax-numpy-bitwise-or "Link to this heading")

jax.numpy.bitwise_or *= \<jnp.ufunc 'bitwise_or'\>*[\#](#jax.numpy.bitwise_or "Link to this definition")  
Compute the bitwise OR operation elementwise.

JAX implementation of [`numpy.bitwise_or`](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_or.html#numpy.bitwise_or "(in NumPy v2.4)"). This is a universal function, and supports the additional APIs described at [`jax.numpy.ufunc`](jax.numpy.ufunc.html#jax.numpy.ufunc "jax.numpy.ufunc"). This function provides the implementation of the `|` operator for JAX arrays.

Parameters:  
- **x** – integer or boolean arrays. Must be broadcastable to a common shape.

- **y** – integer or boolean arrays. Must be broadcastable to a common shape.

- **args** (*ArrayLike*)

- **out** (*None*)

- **where** (*None*)

Returns:  
Array containing the result of the element-wise bitwise OR.

Return type:  
Any

Examples

Calling `bitwise_or` explicitly:

    >>> x = jnp.arange(4)
    >>> jnp.bitwise_or(x, 1)
    Array([1, 1, 3, 3], dtype=int32)

Calling `bitwise_or` via the `|` operator:

    >>> x | 1
    Array([1, 1, 3, 3], dtype=int32)

[](jax.numpy.bitwise_not.html "previous page")

previous

jax.numpy.bitwise_not

[](jax.numpy.bitwise_right_shift.html "next page")

next

jax.numpy.bitwise_right_shift

Contents

- [`bitwise_or`](#jax.numpy.bitwise_or)

By The JAX authors

© Copyright 2024, The JAX Authors.\
