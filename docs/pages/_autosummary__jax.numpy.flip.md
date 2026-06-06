- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.flip

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.flip.rst "Download source file")
-  .pdf

# jax.numpy.flip

## Contents

- [`flip()`](#jax.numpy.flip)

# jax.numpy.flip[\#](#jax-numpy-flip "Link to this heading")

jax.numpy.flip(*m*, *axis=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L1335-L1394)[\#](#jax.numpy.flip "Link to this definition")  
Reverse the order of elements of an array along the given axis.

JAX implementation of [`numpy.flip()`](https://numpy.org/doc/stable/reference/generated/numpy.flip.html#numpy.flip "(in NumPy v2.4)").

Parameters:  
- **m** (*ArrayLike*) – Array.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – integer or sequence of integers. Specifies along which axis or axes should the array elements be reversed. Default is `None`, which flips along all axes.

Returns:  
An array with the elements in reverse order along `axis`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.fliplr()`](jax.numpy.fliplr.html#jax.numpy.fliplr "jax.numpy.fliplr"): reverse the order along axis 1 (left/right)

- [`jax.numpy.flipud()`](jax.numpy.flipud.html#jax.numpy.flipud "jax.numpy.flipud"): reverse the order along axis 0 (up/down)

Examples

    >>> x1 = jnp.array([[1, 2],
    ...                 [3, 4]])
    >>> jnp.flip(x1)
    Array([[4, 3],
           [2, 1]], dtype=int32)

If `axis` is specified with an integer, then `jax.numpy.flip` reverses the array along that particular axis only.

    >>> jnp.flip(x1, axis=1)
    Array([[2, 1],
           [4, 3]], dtype=int32)

    >>> x2 = jnp.arange(1, 9).reshape(2, 2, 2)
    >>> x2
    Array([[[1, 2],
            [3, 4]],

           [[5, 6],
            [7, 8]]], dtype=int32)
    >>> jnp.flip(x2)
    Array([[[8, 7],
            [6, 5]],

           [[4, 3],
            [2, 1]]], dtype=int32)

When `axis` is specified with a sequence of integers, then `jax.numpy.flip` reverses the array along the specified axes.

    >>> jnp.flip(x2, axis=[1, 2])
    Array([[[4, 3],
            [2, 1]],

           [[8, 7],
            [6, 5]]], dtype=int32)

[](jax.numpy.flexible.html "previous page")

previous

jax.numpy.flexible

[](jax.numpy.fliplr.html "next page")

next

jax.numpy.fliplr

Contents

- [`flip()`](#jax.numpy.flip)

By The JAX authors

© Copyright 2024, The JAX Authors.\
