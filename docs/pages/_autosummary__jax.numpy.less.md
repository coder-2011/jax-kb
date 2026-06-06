- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.less

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.less.rst "Download source file")
-  .pdf

# jax.numpy.less

## Contents

- [`less()`](#jax.numpy.less)

# jax.numpy.less[\#](#jax-numpy-less "Link to this heading")

jax.numpy.less(*x*, *y*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L2121-L2166)[\#](#jax.numpy.less "Link to this definition")  
Return element-wise truth value of `x`` ``<`` ``y`.

JAX implementation of [`numpy.less`](https://numpy.org/doc/stable/reference/generated/numpy.less.html#numpy.less "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – input array or scalar.

- **y** (*ArrayLike*) – input array or scalar. `x` and `y` must either have same shape or be broadcast compatible.

Returns:  
An array containing boolean values. `True` if the elements of `x`` ``<`` ``y`, and `False` otherwise.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.greater()`](jax.numpy.greater.html#jax.numpy.greater "jax.numpy.greater"): Returns element-wise truth value of `x`` ``>`` ``y`.

- [`jax.numpy.greater_equal()`](jax.numpy.greater_equal.html#jax.numpy.greater_equal "jax.numpy.greater_equal"): Returns element-wise truth value of `x`` ``>=`` ``y`.

- [`jax.numpy.less_equal()`](jax.numpy.less_equal.html#jax.numpy.less_equal "jax.numpy.less_equal"): Returns element-wise truth value of `x`` ``<=`` ``y`.

Examples

Scalar inputs:

    >>> jnp.less(3, 7)
    Array(True, dtype=bool, weak_type=True)

Inputs with same shape:

    >>> x = jnp.array([5, 9, -3])
    >>> y = jnp.array([1, 6, 4])
    >>> jnp.less(x, y)
    Array([False, False,  True], dtype=bool)

Inputs with broadcast compatibility:

    >>> x1 = jnp.array([[2, -4, 6, -8],
    ...                 [-1, 5, -3, 7]])
    >>> y1 = jnp.array([0, 3, -5, 9])
    >>> jnp.less(x1, y1)
    Array([[False,  True, False,  True],
           [ True, False, False,  True]], dtype=bool)

[](jax.numpy.left_shift.html "previous page")

previous

jax.numpy.left_shift

[](jax.numpy.less_equal.html "next page")

next

jax.numpy.less_equal

Contents

- [`less()`](#jax.numpy.less)

By The JAX authors

© Copyright 2024, The JAX Authors.\
