- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.rollaxis

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.rollaxis.rst "Download source file")
-  .pdf

# jax.numpy.rollaxis

## Contents

- [`rollaxis()`](#jax.numpy.rollaxis)

# jax.numpy.rollaxis[\#](#jax-numpy-rollaxis "Link to this heading")

jax.numpy.rollaxis(*a*, *axis*, *start=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L8550-L8614)[\#](#jax.numpy.rollaxis "Link to this definition")  
Roll the specified axis to a given position.

JAX implementation of [`numpy.rollaxis()`](https://numpy.org/doc/stable/reference/generated/numpy.rollaxis.html#numpy.rollaxis "(in NumPy v2.4)").

This function exists for compatibility with NumPy, but in most cases the newer [`jax.numpy.moveaxis()`](jax.numpy.moveaxis.html#jax.numpy.moveaxis "jax.numpy.moveaxis") instead, because the meaning of its arguments is more intuitive.

Parameters:  
- **a** (*ArrayLike*) – input array.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – index of the axis to roll forward.

- **start** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – index toward which the axis will be rolled (default = 0). After normalizing negative axes, if `start`` ``<=`` ``axis`, the axis is rolled to the `start` index; if `start`` ``>`` ``axis`, the axis is rolled until the position before `start`.

Returns:  
Copy of `a` with rolled axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

Unlike [`numpy.rollaxis()`](https://numpy.org/doc/stable/reference/generated/numpy.rollaxis.html#numpy.rollaxis "(in NumPy v2.4)"), [`jax.numpy.rollaxis()`](#jax.numpy.rollaxis "jax.numpy.rollaxis") will return a copy rather than a view of the input array. However, under JIT, the compiler will optimize away such copies when possible, so this doesn’t have performance impacts in practice.

See also

- [`jax.numpy.moveaxis()`](jax.numpy.moveaxis.html#jax.numpy.moveaxis "jax.numpy.moveaxis"): newer API with clearer semantics than `rollaxis`; this should be preferred to `rollaxis` in most cases.

- [`jax.numpy.swapaxes()`](jax.numpy.swapaxes.html#jax.numpy.swapaxes "jax.numpy.swapaxes"): swap two axes.

- [`jax.numpy.transpose()`](jax.numpy.transpose.html#jax.numpy.transpose "jax.numpy.transpose"): general permutation of axes.

Examples

    >>> a = jnp.ones((2, 3, 4, 5))

Roll axis 2 to the start of the array:

    >>> jnp.rollaxis(a, 2).shape
    (4, 2, 3, 5)

Roll axis 1 to the end of the array:

    >>> jnp.rollaxis(a, 1, a.ndim).shape
    (2, 4, 5, 3)

Equivalent of these two with [`moveaxis()`](jax.numpy.moveaxis.html#jax.numpy.moveaxis "jax.numpy.moveaxis")

    >>> jnp.moveaxis(a, 2, 0).shape
    (4, 2, 3, 5)
    >>> jnp.moveaxis(a, 1, -1).shape
    (2, 4, 5, 3)

[](jax.numpy.roll.html "previous page")

previous

jax.numpy.roll

[](jax.numpy.roots.html "next page")

next

jax.numpy.roots

Contents

- [`rollaxis()`](#jax.numpy.rollaxis)

By The JAX authors

© Copyright 2024, The JAX Authors.\
