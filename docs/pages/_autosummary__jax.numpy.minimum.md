- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.minimum

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.minimum.rst "Download source file")
-  .pdf

# jax.numpy.minimum

## Contents

- [`minimum`](#jax.numpy.minimum)

# jax.numpy.minimum[\#](#jax-numpy-minimum "Link to this heading")

jax.numpy.minimum *= \<jnp.ufunc 'minimum'\>*[\#](#jax.numpy.minimum "Link to this definition")  
Return element-wise minimum of the input arrays.

JAX implementation of [`numpy.minimum`](https://numpy.org/doc/stable/reference/generated/numpy.minimum.html#numpy.minimum "(in NumPy v2.4)").

Parameters:  
- **x** – input array or scalar.

- **y** – input array or scalar. Both `x` and `y` should either have same shape or be broadcast compatible.

- **args** (*ArrayLike*)

- **out** (*None*)

- **where** (*None*)

Returns:  
An array containing the element-wise minimum of `x` and `y`.

Return type:  
Any

Note

For each pair of elements, `jnp.minimum` returns:  
- smaller of the two if both elements are finite numbers.

- `nan` if one element is `nan`.

See also

- [`jax.numpy.maximum()`](jax.numpy.maximum.html#jax.numpy.maximum "jax.numpy.maximum"): Returns element-wise maximum of the input arrays.

- [`jax.numpy.fmin()`](jax.numpy.fmin.html#jax.numpy.fmin "jax.numpy.fmin"): Returns element-wise minimum of the input arrays, ignoring NaNs.

- [`jax.numpy.amin()`](jax.numpy.amin.html#jax.numpy.amin "jax.numpy.amin"): Returns the minimum of array elements along a given axis.

- [`jax.numpy.nanmin()`](jax.numpy.nanmin.html#jax.numpy.nanmin "jax.numpy.nanmin"): Returns the minimum of the array elements along a given axis, ignoring NaNs.

Examples

Inputs with `x.shape`` ``==`` ``y.shape`:

    >>> x = jnp.array([2, 3, 5, 1])
    >>> y = jnp.array([-3, 6, -4, 7])
    >>> jnp.minimum(x, y)
    Array([-3,  3, -4,  1], dtype=int32)

Inputs having broadcast compatibility:

    >>> x1 = jnp.array([[1, 5, 2],
    ...                 [-3, 4, 7]])
    >>> y1 = jnp.array([-2, 3, 6])
    >>> jnp.minimum(x1, y1)
    Array([[-2,  3,  2],
           [-3,  3,  6]], dtype=int32)

Inputs with `nan`:

    >>> nan = jnp.nan
    >>> x2 = jnp.array([[2.5, nan, -2],
    ...                 [nan, 5, 6],
    ...                 [-4, 3, 7]])
    >>> y2 = jnp.array([1, nan, 5])
    >>> jnp.minimum(x2, y2)
    Array([[ 1., nan, -2.],
           [nan, nan,  5.],
           [-4., nan,  5.]], dtype=float32)

[](jax.numpy.min.html "previous page")

previous

jax.numpy.min

[](jax.numpy.mod.html "next page")

next

jax.numpy.mod

Contents

- [`minimum`](#jax.numpy.minimum)

By The JAX authors

© Copyright 2024, The JAX Authors.\
