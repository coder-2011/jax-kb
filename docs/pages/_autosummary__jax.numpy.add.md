- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.add

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.add.rst "Download source file")
-  .pdf

# jax.numpy.add

## Contents

- [`add`](#jax.numpy.add)

# jax.numpy.add[\#](#jax-numpy-add "Link to this heading")

jax.numpy.add *= \<jnp.ufunc 'add'\>*[\#](#jax.numpy.add "Link to this definition")  
Add two arrays element-wise.

JAX implementation of [`numpy.add`](https://numpy.org/doc/stable/reference/generated/numpy.add.html#numpy.add "(in NumPy v2.4)"). This is a universal function, and supports the additional APIs described at [`jax.numpy.ufunc`](jax.numpy.ufunc.html#jax.numpy.ufunc "jax.numpy.ufunc"). This function provides the implementation of the `+` operator for JAX arrays.

Parameters:  
- **x** – arrays to add. Must be broadcastable to a common shape.

- **y** – arrays to add. Must be broadcastable to a common shape.

- **args** (*ArrayLike*)

- **out** (*None*)

- **where** (*None*)

Returns:  
Array containing the result of the element-wise addition.

Return type:  
Any

Examples

Calling `add` explicitly:

    >>> x = jnp.arange(4)
    >>> jnp.add(x, 10)
    Array([10, 11, 12, 13], dtype=int32)

Calling `add` via the `+` operator:

    >>> x + 10
    Array([10, 11, 12, 13], dtype=int32)

[](jax.numpy.acosh.html "previous page")

previous

jax.numpy.acosh

[](jax.numpy.all.html "next page")

next

jax.numpy.all

Contents

- [`add`](#jax.numpy.add)

By The JAX authors

© Copyright 2024, The JAX Authors.\
