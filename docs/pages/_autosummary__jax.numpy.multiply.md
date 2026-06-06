- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.multiply

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.multiply.rst "Download source file")
-  .pdf

# jax.numpy.multiply

## Contents

- [`multiply`](#jax.numpy.multiply)

# jax.numpy.multiply[\#](#jax-numpy-multiply "Link to this heading")

jax.numpy.multiply *= \<jnp.ufunc 'multiply'\>*[\#](#jax.numpy.multiply "Link to this definition")  
Multiply two arrays element-wise.

JAX implementation of [`numpy.multiply`](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html#numpy.multiply "(in NumPy v2.4)"). This is a universal function, and supports the additional APIs described at [`jax.numpy.ufunc`](jax.numpy.ufunc.html#jax.numpy.ufunc "jax.numpy.ufunc"). This function provides the implementation of the `*` operator for JAX arrays.

Parameters:  
- **x** – arrays to multiply. Must be broadcastable to a common shape.

- **y** – arrays to multiply. Must be broadcastable to a common shape.

- **args** (*ArrayLike*)

- **out** (*None*)

- **where** (*None*)

Returns:  
Array containing the result of the element-wise multiplication.

Return type:  
Any

Examples

Calling `multiply` explicitly:

    >>> x = jnp.arange(4)
    >>> jnp.multiply(x, 10)
    Array([ 0, 10, 20, 30], dtype=int32)

Calling `multiply` via the `*` operator:

    >>> x * 10
    Array([ 0, 10, 20, 30], dtype=int32)

[](jax.numpy.moveaxis.html "previous page")

previous

jax.numpy.moveaxis

[](jax.numpy.nan_to_num.html "next page")

next

jax.numpy.nan_to_num

Contents

- [`multiply`](#jax.numpy.multiply)

By The JAX authors

© Copyright 2024, The JAX Authors.\
