- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.not_equal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.not_equal.rst "Download source file")
-  .pdf

# jax.numpy.not_equal

## Contents

- [`not_equal()`](#jax.numpy.not_equal)

# jax.numpy.not_equal[\#](#jax-numpy-not-equal "Link to this heading")

jax.numpy.not_equal(*x*, *y*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L1487-L1535)[\#](#jax.numpy.not_equal "Link to this definition")  
Returns element-wise truth value of `x`` ``!=`` ``y`.

JAX implementation of [`numpy.not_equal`](https://numpy.org/doc/stable/reference/generated/numpy.not_equal.html#numpy.not_equal "(in NumPy v2.4)"). This function provides the implementation of the `!=` operator for JAX arrays.

Parameters:  
- **x** (*ArrayLike*) – input array or scalar.

- **y** (*ArrayLike*) – input array or scalar. `x` and `y` should either have same shape or be broadcast compatible.

Returns:  
A boolean array containing `True` where the elements of `x`` ``!=`` ``y` and `False` otherwise.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.equal()`](jax.numpy.equal.html#jax.numpy.equal "jax.numpy.equal"): Returns element-wise truth value of `x`` ``==`` ``y`.

- [`jax.numpy.greater_equal()`](jax.numpy.greater_equal.html#jax.numpy.greater_equal "jax.numpy.greater_equal"): Returns element-wise truth value of `x`` ``>=`` ``y`.

- [`jax.numpy.less_equal()`](jax.numpy.less_equal.html#jax.numpy.less_equal "jax.numpy.less_equal"): Returns element-wise truth value of `x`` ``<=`` ``y`.

- [`jax.numpy.greater()`](jax.numpy.greater.html#jax.numpy.greater "jax.numpy.greater"): Returns element-wise truth value of `x`` ``>`` ``y`.

- [`jax.numpy.less()`](jax.numpy.less.html#jax.numpy.less "jax.numpy.less"): Returns element-wise truth value of `x`` ``<`` ``y`.

Examples

    >>> jnp.not_equal(0., -0.)
    Array(False, dtype=bool, weak_type=True)
    >>> jnp.not_equal(-2, 2)
    Array(True, dtype=bool, weak_type=True)
    >>> jnp.not_equal(1, 1.)
    Array(False, dtype=bool, weak_type=True)
    >>> jnp.not_equal(5, jnp.array(5))
    Array(False, dtype=bool, weak_type=True)
    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6],
    ...                [7, 8, 9]])
    >>> y = jnp.array([1, 5, 9])
    >>> jnp.not_equal(x, y)
    Array([[False,  True,  True],
           [ True, False,  True],
           [ True,  True, False]], dtype=bool)
    >>> x != y
    Array([[False,  True,  True],
           [ True, False,  True],
           [ True,  True, False]], dtype=bool)

[](jax.numpy.nonzero.html "previous page")

previous

jax.numpy.nonzero

[](jax.numpy.number.html "next page")

next

jax.numpy.number

Contents

- [`not_equal()`](#jax.numpy.not_equal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
