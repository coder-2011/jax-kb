- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.subtract

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.subtract.rst "Download source file")
-  .pdf

# jax.numpy.subtract

## Contents

- [`subtract`](#jax.numpy.subtract)

# jax.numpy.subtract[\#](#jax-numpy-subtract "Link to this heading")

jax.numpy.subtract *= \<jnp.ufunc 'subtract'\>*[\#](#jax.numpy.subtract "Link to this definition")  
Subtract two arrays element-wise.

JAX implementation of [`numpy.subtract`](https://numpy.org/doc/stable/reference/generated/numpy.subtract.html#numpy.subtract "(in NumPy v2.4)"). This is a universal function, and supports the additional APIs described at [`jax.numpy.ufunc`](jax.numpy.ufunc.html#jax.numpy.ufunc "jax.numpy.ufunc"). This function provides the implementation of the `-` operator for JAX arrays.

Parameters:  
- **x** – arrays to subtract. Must be broadcastable to a common shape.

- **y** – arrays to subtract. Must be broadcastable to a common shape.

- **args** (*ArrayLike*)

- **out** (*None*)

- **where** (*None*)

Returns:  
Array containing the result of the element-wise subtraction.

Return type:  
Any

Examples

Calling `subtract` explicitly:

    >>> x = jnp.arange(4)
    >>> jnp.subtract(x, 10)
    Array([-10,  -9,  -8,  -7], dtype=int32)

Calling `subtract` via the `-` operator:

    >>> x - 10
    Array([-10,  -9,  -8,  -7], dtype=int32)

[](jax.numpy.std.html "previous page")

previous

jax.numpy.std

[](jax.numpy.sum.html "next page")

next

jax.numpy.sum

Contents

- [`subtract`](#jax.numpy.subtract)

By The JAX authors

© Copyright 2024, The JAX Authors.\
