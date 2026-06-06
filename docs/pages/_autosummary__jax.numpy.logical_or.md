- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.logical_or

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.logical_or.rst "Download source file")
-  .pdf

# jax.numpy.logical_or

## Contents

- [`logical_or`](#jax.numpy.logical_or)

# jax.numpy.logical_or[\#](#jax-numpy-logical-or "Link to this heading")

jax.numpy.logical_or *= \<jnp.ufunc 'logical_or'\>*[\#](#jax.numpy.logical_or "Link to this definition")  
Compute the logical OR operation elementwise.

JAX implementation of [`numpy.logical_or`](https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html#numpy.logical_or "(in NumPy v2.4)"). This is a universal function, and supports the additional APIs described at [`jax.numpy.ufunc`](jax.numpy.ufunc.html#jax.numpy.ufunc "jax.numpy.ufunc").

Parameters:  
- **x** – input arrays. Must be broadcastable to a common shape.

- **y** – input arrays. Must be broadcastable to a common shape.

- **args** (*ArrayLike*)

- **out** (*None*)

- **where** (*None*)

Returns:  
Array containing the result of the element-wise logical OR.

Return type:  
Any

Examples

    >>> x = jnp.arange(4)
    >>> jnp.logical_or(x, 1)
    Array([ True,  True,  True,  True], dtype=bool)

[](jax.numpy.logical_not.html "previous page")

previous

jax.numpy.logical_not

[](jax.numpy.logical_xor.html "next page")

next

jax.numpy.logical_xor

Contents

- [`logical_or`](#jax.numpy.logical_or)

By The JAX authors

© Copyright 2024, The JAX Authors.\
