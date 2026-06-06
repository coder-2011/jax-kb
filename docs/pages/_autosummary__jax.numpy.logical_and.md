- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.logical_and

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.logical_and.rst "Download source file")
-  .pdf

# jax.numpy.logical_and

## Contents

- [`logical_and`](#jax.numpy.logical_and)

# jax.numpy.logical_and[\#](#jax-numpy-logical-and "Link to this heading")

jax.numpy.logical_and *= \<jnp.ufunc 'logical_and'\>*[\#](#jax.numpy.logical_and "Link to this definition")  
Compute the logical AND operation elementwise.

JAX implementation of [`numpy.logical_and`](https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html#numpy.logical_and "(in NumPy v2.4)"). This is a universal function, and supports the additional APIs described at [`jax.numpy.ufunc`](jax.numpy.ufunc.html#jax.numpy.ufunc "jax.numpy.ufunc").

Parameters:  
- **x** – input arrays. Must be broadcastable to a common shape.

- **y** – input arrays. Must be broadcastable to a common shape.

- **args** (*ArrayLike*)

- **out** (*None*)

- **where** (*None*)

Returns:  
Array containing the result of the element-wise logical AND.

Return type:  
Any

Examples

    >>> x = jnp.arange(4)
    >>> jnp.logical_and(x, 1)
    Array([False,  True,  True,  True], dtype=bool)

[](jax.numpy.logaddexp2.html "previous page")

previous

jax.numpy.logaddexp2

[](jax.numpy.logical_not.html "next page")

next

jax.numpy.logical_not

Contents

- [`logical_and`](#jax.numpy.logical_and)

By The JAX authors

© Copyright 2024, The JAX Authors.\
