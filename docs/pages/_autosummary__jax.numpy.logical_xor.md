- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.logical_xor

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.logical_xor.rst "Download source file")
-  .pdf

# jax.numpy.logical_xor

## Contents

- [`logical_xor`](#jax.numpy.logical_xor)

# jax.numpy.logical_xor[\#](#jax-numpy-logical-xor "Link to this heading")

jax.numpy.logical_xor *= \<jnp.ufunc 'logical_xor'\>*[\#](#jax.numpy.logical_xor "Link to this definition")  
Compute the logical XOR operation elementwise.

JAX implementation of [`numpy.logical_xor`](https://numpy.org/doc/stable/reference/generated/numpy.logical_xor.html#numpy.logical_xor "(in NumPy v2.4)"). This is a universal function, and supports the additional APIs described at [`jax.numpy.ufunc`](jax.numpy.ufunc.html#jax.numpy.ufunc "jax.numpy.ufunc").

Parameters:  
- **x** – input arrays. Must be broadcastable to a common shape.

- **y** – input arrays. Must be broadcastable to a common shape.

- **args** (*ArrayLike*)

- **out** (*None*)

- **where** (*None*)

Returns:  
Array containing the result of the element-wise logical XOR.

Return type:  
Any

Examples

    >>> x = jnp.arange(4)
    >>> jnp.logical_xor(x, 1)
    Array([ True, False, False, False], dtype=bool)

[](jax.numpy.logical_or.html "previous page")

previous

jax.numpy.logical_or

[](jax.numpy.logspace.html "next page")

next

jax.numpy.logspace

Contents

- [`logical_xor`](#jax.numpy.logical_xor)

By The JAX authors

© Copyright 2024, The JAX Authors.\
