- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.rot90

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.rot90.rst "Download source file")
-  .pdf

# jax.numpy.rot90

## Contents

- [`rot90()`](#jax.numpy.rot90)

# jax.numpy.rot90[\#](#jax-numpy-rot90 "Link to this heading")

jax.numpy.rot90(*m*, *k=1*, *axes=(0, 1)*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L1251-L1333)[\#](#jax.numpy.rot90 "Link to this definition")  
Rotate an array by 90 degrees counterclockwise in the plane specified by axes.

JAX implementation of [`numpy.rot90()`](https://numpy.org/doc/stable/reference/generated/numpy.rot90.html#numpy.rot90 "(in NumPy v2.4)").

Parameters:  
- **m** (*ArrayLike*) – input array. Must have `m.ndim`` ``>=`` ``2`.

- **k** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int, optional, default=1. Specifies the number of times the array is rotated. For negative values of `k`, the array is rotated in clockwise direction.

- **axes** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – tuple of 2 integers, optional, default= (0, 1). The axes define the plane in which the array is rotated. Both the axes must be different.

Returns:  
An array containing the copy of the input, `m` rotated by 90 degrees.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.flip()`](jax.numpy.flip.html#jax.numpy.flip "jax.numpy.flip"): reverse the order along the given axis

- [`jax.numpy.fliplr()`](jax.numpy.fliplr.html#jax.numpy.fliplr "jax.numpy.fliplr"): reverse the order along axis 1 (left/right)

- [`jax.numpy.flipud()`](jax.numpy.flipud.html#jax.numpy.flipud "jax.numpy.flipud"): reverse the order along axis 0 (up/down)

Examples

    >>> m = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> jnp.rot90(m)
    Array([[3, 6],
           [2, 5],
           [1, 4]], dtype=int32)
    >>> jnp.rot90(m, k=2)
    Array([[6, 5, 4],
           [3, 2, 1]], dtype=int32)

`jnp.rot90(m,`` ``k=1,`` ``axes=(1,`` ``0))` is equivalent to `jnp.rot90(m,`` ``k=-1,`` ``axes(0,1))`.

    >>> jnp.rot90(m, axes=(1, 0))
    Array([[4, 1],
           [5, 2],
           [6, 3]], dtype=int32)
    >>> jnp.rot90(m, k=-1, axes=(0, 1))
    Array([[4, 1],
           [5, 2],
           [6, 3]], dtype=int32)

when input array has `ndim>2`:

    >>> m1 = jnp.array([[[1, 2, 3],
    ...                  [4, 5, 6]],
    ...                 [[7, 8, 9],
    ...                  [10, 11, 12]]])
    >>> jnp.rot90(m1, k=1, axes=(2, 1))
    Array([[[ 4,  1],
            [ 5,  2],
            [ 6,  3]],

           [[10,  7],
            [11,  8],
            [12,  9]]], dtype=int32)

[](jax.numpy.roots.html "previous page")

previous

jax.numpy.roots

[](jax.numpy.round.html "next page")

next

jax.numpy.round

Contents

- [`rot90()`](#jax.numpy.rot90)

By The JAX authors

© Copyright 2024, The JAX Authors.\
