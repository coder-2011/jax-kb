- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.less_equal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.less_equal.rst "Download source file")
-  .pdf

# jax.numpy.less_equal

## Contents

- [`less_equal()`](#jax.numpy.less_equal)

# jax.numpy.less_equal[\#](#jax-numpy-less-equal "Link to this heading")

jax.numpy.less_equal(*x*, *y*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L2074-L2119)[\#](#jax.numpy.less_equal "Link to this definition")  
Return element-wise truth value of `x`` ``<=`` ``y`.

JAX implementation of [`numpy.less_equal`](https://numpy.org/doc/stable/reference/generated/numpy.less_equal.html#numpy.less_equal "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – input array or scalar.

- **y** (*ArrayLike*) – input array or scalar. `x` and `y` must have either same shape or be broadcast compatible.

Returns:  
An array containing the boolean values. `True` if the elements of `x`` ``<=`` ``y`, and `False` otherwise.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.greater_equal()`](jax.numpy.greater_equal.html#jax.numpy.greater_equal "jax.numpy.greater_equal"): Returns element-wise truth value of `x`` ``>=`` ``y`.

- [`jax.numpy.greater()`](jax.numpy.greater.html#jax.numpy.greater "jax.numpy.greater"): Returns element-wise truth value of `x`` ``>`` ``y`.

- [`jax.numpy.less()`](jax.numpy.less.html#jax.numpy.less "jax.numpy.less"): Returns element-wise truth value of `x`` ``<`` ``y`.

Examples

Scalar inputs:

    >>> jnp.less_equal(6, -2)
    Array(False, dtype=bool, weak_type=True)

Inputs with same shape:

    >>> x = jnp.array([-4, 1, 7])
    >>> y = jnp.array([2, -3, 8])
    >>> jnp.less_equal(x, y)
    Array([ True, False,  True], dtype=bool)

Inputs with broadcast compatibility:

    >>> x1 = jnp.array([2, -5, 9])
    >>> y1 = jnp.array([[1, -6, 5],
    ...                 [-2, 4, -6]])
    >>> jnp.less_equal(x1, y1)
    Array([[False, False, False],
           [False,  True, False]], dtype=bool)

[](jax.numpy.less.html "previous page")

previous

jax.numpy.less

[](jax.numpy.lexsort.html "next page")

next

jax.numpy.lexsort

Contents

- [`less_equal()`](#jax.numpy.less_equal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
