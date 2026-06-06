- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.flipud

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.flipud.rst "Download source file")
-  .pdf

# jax.numpy.flipud

## Contents

- [`flipud()`](#jax.numpy.flipud)

# jax.numpy.flipud[\#](#jax-numpy-flipud "Link to this heading")

jax.numpy.flipud(*m*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L1430-L1455)[\#](#jax.numpy.flipud "Link to this definition")  
Reverse the order of elements of an array along axis 0.

JAX implementation of [`numpy.flipud()`](https://numpy.org/doc/stable/reference/generated/numpy.flipud.html#numpy.flipud "(in NumPy v2.4)").

Parameters:  
**m** (*ArrayLike*) – Array with at least one dimension.

Returns:  
An array with the elements in reverse order along axis 0.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.flip()`](jax.numpy.flip.html#jax.numpy.flip "jax.numpy.flip"): reverse the order along the given axis

- [`jax.numpy.fliplr()`](jax.numpy.fliplr.html#jax.numpy.fliplr "jax.numpy.fliplr"): reverse the order along axis 1

Examples

    >>> x = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> jnp.flipud(x)
    Array([[3, 4],
           [1, 2]], dtype=int32)

[](jax.numpy.fliplr.html "previous page")

previous

jax.numpy.fliplr

[](jax.numpy.float_.html "next page")

next

jax.numpy.float\_

Contents

- [`flipud()`](#jax.numpy.flipud)

By The JAX authors

© Copyright 2024, The JAX Authors.\
