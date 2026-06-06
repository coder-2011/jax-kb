- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.swapaxes

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.swapaxes.rst "Download source file")
-  .pdf

# jax.numpy.swapaxes

## Contents

- [`swapaxes()`](#jax.numpy.swapaxes)

# jax.numpy.swapaxes[\#](#jax-numpy-swapaxes "Link to this heading")

jax.numpy.swapaxes(*a*, *axis1*, *axis2*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L2411-L2457)[\#](#jax.numpy.swapaxes "Link to this definition")  
Swap two axes of an array.

JAX implementation of [`numpy.swapaxes()`](https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html#numpy.swapaxes "(in NumPy v2.4)"), implemented in terms of [`jax.lax.transpose()`](jax.lax.transpose.html#jax.lax.transpose "jax.lax.transpose").

Parameters:  
- **a** (*ArrayLike*) – input array

- **axis1** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – index of first axis

- **axis2** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – index of second axis

Returns:  
Copy of `a` with specified axes swapped.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

Unlike [`numpy.swapaxes()`](https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html#numpy.swapaxes "(in NumPy v2.4)"), [`jax.numpy.swapaxes()`](#jax.numpy.swapaxes "jax.numpy.swapaxes") will return a copy rather than a view of the input array. However, under JIT, the compiler will optimize away such copies when possible, so this doesn’t have performance impacts in practice.

See also

- [`jax.numpy.moveaxis()`](jax.numpy.moveaxis.html#jax.numpy.moveaxis "jax.numpy.moveaxis"): move a single axis of an array.

- [`jax.numpy.rollaxis()`](jax.numpy.rollaxis.html#jax.numpy.rollaxis "jax.numpy.rollaxis"): older API for `moveaxis`.

- [`jax.lax.transpose()`](jax.lax.transpose.html#jax.lax.transpose "jax.lax.transpose"): more general axes permutations.

- [`jax.Array.swapaxes()`](jax.Array.swapaxes.html#jax.Array.swapaxes "jax.Array.swapaxes"): same functionality via an array method.

Examples

    >>> a = jnp.ones((2, 3, 4, 5))
    >>> jnp.swapaxes(a, 1, 3).shape
    (2, 5, 4, 3)

Equivalent output via the `swapaxes` array method:

    >>> a.swapaxes(1, 3).shape
    (2, 5, 4, 3)

Equivalent output via [`transpose()`](jax.numpy.transpose.html#jax.numpy.transpose "jax.numpy.transpose"):

    >>> a.transpose(0, 3, 2, 1).shape
    (2, 5, 4, 3)

[](jax.numpy.sum.html "previous page")

previous

jax.numpy.sum

[](jax.numpy.take.html "next page")

next

jax.numpy.take

Contents

- [`swapaxes()`](#jax.numpy.swapaxes)

By The JAX authors

© Copyright 2024, The JAX Authors.\
