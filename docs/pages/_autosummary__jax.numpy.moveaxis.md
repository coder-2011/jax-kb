- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.moveaxis

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.moveaxis.rst "Download source file")
-  .pdf

# jax.numpy.moveaxis

## Contents

- [`moveaxis()`](#jax.numpy.moveaxis)

# jax.numpy.moveaxis[\#](#jax-numpy-moveaxis "Link to this heading")

jax.numpy.moveaxis(*a*, *source*, *destination*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L2459-L2511)[\#](#jax.numpy.moveaxis "Link to this definition")  
Move an array axis to a new position

JAX implementation of [`numpy.moveaxis()`](https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html#numpy.moveaxis "(in NumPy v2.4)"), implemented in terms of [`jax.lax.transpose()`](jax.lax.transpose.html#jax.lax.transpose "jax.lax.transpose").

Parameters:  
- **a** (*ArrayLike*) – input array

- **source** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – index or indices of the axes to move.

- **destination** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – index or indices of the axes destinations

Returns:  
Copy of `a` with axes moved from `source` to `destination`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

Unlike [`numpy.moveaxis()`](https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html#numpy.moveaxis "(in NumPy v2.4)"), [`jax.numpy.moveaxis()`](#jax.numpy.moveaxis "jax.numpy.moveaxis") will return a copy rather than a view of the input array. However, under JIT, the compiler will optimize away such copies when possible, so this doesn’t have performance impacts in practice.

See also

- [`jax.numpy.swapaxes()`](jax.numpy.swapaxes.html#jax.numpy.swapaxes "jax.numpy.swapaxes"): swap two axes.

- [`jax.numpy.rollaxis()`](jax.numpy.rollaxis.html#jax.numpy.rollaxis "jax.numpy.rollaxis"): older API for moving an axis.

- [`jax.numpy.transpose()`](jax.numpy.transpose.html#jax.numpy.transpose "jax.numpy.transpose"): general axes permutation.

Examples

    >>> a = jnp.ones((2, 3, 4, 5))

Move axis `1` to the end of the array:

    >>> jnp.moveaxis(a, 1, -1).shape
    (2, 4, 5, 3)

Move the last axis to position 1:

    >>> jnp.moveaxis(a, -1, 1).shape
    (2, 5, 3, 4)

Move multiple axes:

    >>> jnp.moveaxis(a, (0, 1), (-1, -2)).shape
    (4, 5, 3, 2)

This can also be accomplished via [`transpose()`](jax.numpy.transpose.html#jax.numpy.transpose "jax.numpy.transpose"):

    >>> a.transpose(2, 3, 1, 0).shape
    (4, 5, 3, 2)

[](jax.numpy.modf.html "previous page")

previous

jax.numpy.modf

[](jax.numpy.multiply.html "next page")

next

jax.numpy.multiply

Contents

- [`moveaxis()`](#jax.numpy.moveaxis)

By The JAX authors

© Copyright 2024, The JAX Authors.\
