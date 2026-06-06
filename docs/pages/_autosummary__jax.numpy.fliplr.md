- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fliplr

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fliplr.rst "Download source file")
-  .pdf

# jax.numpy.fliplr

## Contents

- [`fliplr()`](#jax.numpy.fliplr)

# jax.numpy.fliplr[\#](#jax-numpy-fliplr "Link to this heading")

jax.numpy.fliplr(*m*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L1403-L1428)[\#](#jax.numpy.fliplr "Link to this definition")  
Reverse the order of elements of an array along axis 1.

JAX implementation of [`numpy.fliplr()`](https://numpy.org/doc/stable/reference/generated/numpy.fliplr.html#numpy.fliplr "(in NumPy v2.4)").

Parameters:  
**m** (*ArrayLike*) – Array with at least two dimensions.

Returns:  
An array with the elements in reverse order along axis 1.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.flip()`](jax.numpy.flip.html#jax.numpy.flip "jax.numpy.flip"): reverse the order along the given axis

- [`jax.numpy.flipud()`](jax.numpy.flipud.html#jax.numpy.flipud "jax.numpy.flipud"): reverse the order along axis 0

Examples

    >>> x = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> jnp.fliplr(x)
    Array([[2, 1],
           [4, 3]], dtype=int32)

[](jax.numpy.flip.html "previous page")

previous

jax.numpy.flip

[](jax.numpy.flipud.html "next page")

next

jax.numpy.flipud

Contents

- [`fliplr()`](#jax.numpy.fliplr)

By The JAX authors

© Copyright 2024, The JAX Authors.\
