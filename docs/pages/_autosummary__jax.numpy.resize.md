- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.resize

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.resize.rst "Download source file")
-  .pdf

# jax.numpy.resize

## Contents

- [`resize()`](#jax.numpy.resize)

# jax.numpy.resize[\#](#jax-numpy-resize "Link to this heading")

jax.numpy.resize(*a*, *new_shape*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L2217-L2267)[\#](#jax.numpy.resize "Link to this definition")  
Return a new array with specified shape.

JAX implementation of [`numpy.resize()`](https://numpy.org/doc/stable/reference/generated/numpy.resize.html#numpy.resize "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array or scalar.

- **new_shape** (*Shape*) – int or tuple of ints. Specifies the shape of the resized array.

Returns:  
A resized array with specified shape. The elements of `a` are repeated in the resized array, if the resized array is larger than the original array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.reshape()`](jax.numpy.reshape.html#jax.numpy.reshape "jax.numpy.reshape"): Returns a reshaped copy of an array.

- [`jax.numpy.repeat()`](jax.numpy.repeat.html#jax.numpy.repeat "jax.numpy.repeat"): Constructs an array from repeated elements.

Examples

    >>> x = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> jnp.resize(x, (3, 3))
    Array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]], dtype=int32)
    >>> jnp.resize(x, (3, 4))
    Array([[1, 2, 3, 4],
           [5, 6, 7, 8],
           [9, 1, 2, 3]], dtype=int32)
    >>> jnp.resize(4, (3, 2))
    Array([[4, 4],
           [4, 4],
           [4, 4]], dtype=int32, weak_type=True)

[](jax.numpy.reshape.html "previous page")

previous

jax.numpy.reshape

[](jax.numpy.result_type.html "next page")

next

jax.numpy.result_type

Contents

- [`resize()`](#jax.numpy.resize)

By The JAX authors

© Copyright 2024, The JAX Authors.\
