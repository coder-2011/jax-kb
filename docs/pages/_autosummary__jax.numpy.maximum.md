- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.maximum

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.maximum.rst "Download source file")
-  .pdf

# jax.numpy.maximum

## Contents

- [`maximum`](#jax.numpy.maximum)

# jax.numpy.maximum[\#](#jax-numpy-maximum "Link to this heading")

jax.numpy.maximum *= \<jnp.ufunc 'maximum'\>*[\#](#jax.numpy.maximum "Link to this definition")  
Return element-wise maximum of the input arrays.

JAX implementation of [`numpy.maximum`](https://numpy.org/doc/stable/reference/generated/numpy.maximum.html#numpy.maximum "(in NumPy v2.4)").

Parameters:  
- **x** – input array or scalar.

- **y** – input array or scalar. Both `x` and `y` should either have same shape or be broadcast compatible.

- **args** (*ArrayLike*)

- **out** (*None*)

- **where** (*None*)

Returns:  
An array containing the element-wise maximum of `x` and `y`.

Return type:  
Any

Note

For each pair of elements, `jnp.maximum` returns:  
- larger of the two if both elements are finite numbers.

- `nan` if one element is `nan`.

See also

- [`jax.numpy.minimum()`](jax.numpy.minimum.html#jax.numpy.minimum "jax.numpy.minimum"): Returns element-wise minimum of the input arrays.

- [`jax.numpy.fmax()`](jax.numpy.fmax.html#jax.numpy.fmax "jax.numpy.fmax"): Returns element-wise maximum of the input arrays, ignoring NaNs.

- [`jax.numpy.amax()`](jax.numpy.amax.html#jax.numpy.amax "jax.numpy.amax"): Returns the maximum of array elements along a given axis.

- [`jax.numpy.nanmax()`](jax.numpy.nanmax.html#jax.numpy.nanmax "jax.numpy.nanmax"): Returns the maximum of the array elements along a given axis, ignoring NaNs.

Examples

Inputs with `x.shape`` ``==`` ``y.shape`:

    >>> x = jnp.array([1, -5, 3, 2])
    >>> y = jnp.array([-2, 4, 7, -6])
    >>> jnp.maximum(x, y)
    Array([1, 4, 7, 2], dtype=int32)

Inputs with broadcast compatibility:

    >>> x1 = jnp.array([[-2, 5, 7, 4],
    ...                 [1, -6, 3, 8]])
    >>> y1 = jnp.array([-5, 3, 6, 9])
    >>> jnp.maximum(x1, y1)
    Array([[-2,  5,  7,  9],
           [ 1,  3,  6,  9]], dtype=int32)

Inputs having `nan`:

    >>> nan = jnp.nan
    >>> x2 = jnp.array([nan, -3, 9])
    >>> y2 = jnp.array([[4, -2, nan],
    ...                 [-3, -5, 10]])
    >>> jnp.maximum(x2, y2)
    Array([[nan, -2., nan],
          [nan, -3., 10.]], dtype=float32)

[](jax.numpy.max.html "previous page")

previous

jax.numpy.max

[](jax.numpy.mean.html "next page")

next

jax.numpy.mean

Contents

- [`maximum`](#jax.numpy.maximum)

By The JAX authors

© Copyright 2024, The JAX Authors.\
