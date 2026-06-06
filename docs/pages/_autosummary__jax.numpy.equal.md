- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.equal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.equal.rst "Download source file")
-  .pdf

# jax.numpy.equal

## Contents

- [`equal()`](#jax.numpy.equal)

# jax.numpy.equal[\#](#jax-numpy-equal "Link to this heading")

jax.numpy.equal(*x*, *y*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L1437-L1485)[\#](#jax.numpy.equal "Link to this definition")  
Returns element-wise truth value of `x`` ``==`` ``y`.

JAX implementation of [`numpy.equal`](https://numpy.org/doc/stable/reference/generated/numpy.equal.html#numpy.equal "(in NumPy v2.4)"). This function provides the implementation of the `==` operator for JAX arrays.

Parameters:  
- **x** (*ArrayLike*) – input array or scalar.

- **y** (*ArrayLike*) – input array or scalar. `x` and `y` should either have same shape or be broadcast compatible.

Returns:  
A boolean array containing `True` where the elements of `x`` ``==`` ``y` and `False` otherwise.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.not_equal()`](jax.numpy.not_equal.html#jax.numpy.not_equal "jax.numpy.not_equal"): Returns element-wise truth value of `x`` ``!=`` ``y`.

- [`jax.numpy.greater_equal()`](jax.numpy.greater_equal.html#jax.numpy.greater_equal "jax.numpy.greater_equal"): Returns element-wise truth value of `x`` ``>=`` ``y`.

- [`jax.numpy.less_equal()`](jax.numpy.less_equal.html#jax.numpy.less_equal "jax.numpy.less_equal"): Returns element-wise truth value of `x`` ``<=`` ``y`.

- [`jax.numpy.greater()`](jax.numpy.greater.html#jax.numpy.greater "jax.numpy.greater"): Returns element-wise truth value of `x`` ``>`` ``y`.

- [`jax.numpy.less()`](jax.numpy.less.html#jax.numpy.less "jax.numpy.less"): Returns element-wise truth value of `x`` ``<`` ``y`.

Examples

    >>> jnp.equal(0., -0.)
    Array(True, dtype=bool, weak_type=True)
    >>> jnp.equal(1, 1.)
    Array(True, dtype=bool, weak_type=True)
    >>> jnp.equal(5, jnp.array(5))
    Array(True, dtype=bool, weak_type=True)
    >>> jnp.equal(2, -2)
    Array(False, dtype=bool, weak_type=True)
    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6],
    ...                [7, 8, 9]])
    >>> y = jnp.array([1, 5, 9])
    >>> jnp.equal(x, y)
    Array([[ True, False, False],
           [False,  True, False],
           [False, False,  True]], dtype=bool)
    >>> x == y
    Array([[ True, False, False],
           [False,  True, False],
           [False, False,  True]], dtype=bool)

[](jax.numpy.empty_like.html "previous page")

previous

jax.numpy.empty_like

[](jax.numpy.exp.html "next page")

next

jax.numpy.exp

Contents

- [`equal()`](#jax.numpy.equal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
