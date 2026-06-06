- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.floor_divide

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.floor_divide.rst "Download source file")
-  .pdf

# jax.numpy.floor_divide

## Contents

- [`floor_divide()`](#jax.numpy.floor_divide)

# jax.numpy.floor_divide[\#](#jax-numpy-floor-divide "Link to this heading")

jax.numpy.floor_divide(*x1*, *x2*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L2489-L2543)[\#](#jax.numpy.floor_divide "Link to this definition")  
Calculates the floor division of x1 by x2 element-wise

JAX implementation of [`numpy.floor_divide`](https://numpy.org/doc/stable/reference/generated/numpy.floor_divide.html#numpy.floor_divide "(in NumPy v2.4)").

Parameters:  
- **x1** (*ArrayLike*) – Input array, the dividend

- **x2** (*ArrayLike*) – Input array, the divisor

Returns:  
An array-like object containing each of the quotients rounded down to the nearest integer towards negative infinity. This is equivalent to `x1`` ``//`` ``x2` in Python.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

`x1`` ``//`` ``x2` is equivalent to `jnp.floor_divide(x1,`` ``x2)` for arrays `x1` and `x2`

See also

[`jax.numpy.divide()`](jax.numpy.divide.html#jax.numpy.divide "jax.numpy.divide") and [`jax.numpy.true_divide()`](jax.numpy.true_divide.html#jax.numpy.true_divide "jax.numpy.true_divide") for floating point division.

Examples

    >>> x1 = jnp.array([10, 20, 30])
    >>> x2 = jnp.array([3, 4, 7])
    >>> jnp.floor_divide(x1, x2)
    Array([3, 5, 4], dtype=int32)

    >>> x1 = jnp.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    >>> x2 = 3
    >>> jnp.floor_divide(x1, x2)
    Array([-2, -2, -1, -1, -1,  0,  0,  0,  1,  1,  1], dtype=int32)

    >>> x1 = jnp.array([6, 6, 6], dtype=jnp.int32)
    >>> x2 = jnp.array([2.0, 2.5, 3.0], dtype=jnp.float32)
    >>> jnp.floor_divide(x1, x2)
    Array([3., 2., 2.], dtype=float32)

[](jax.numpy.floor.html "previous page")

previous

jax.numpy.floor

[](jax.numpy.fmax.html "next page")

next

jax.numpy.fmax

Contents

- [`floor_divide()`](#jax.numpy.floor_divide)

By The JAX authors

© Copyright 2024, The JAX Authors.\
