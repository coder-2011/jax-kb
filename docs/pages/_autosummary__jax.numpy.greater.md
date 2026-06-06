- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.greater

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.greater.rst "Download source file")
-  .pdf

# jax.numpy.greater

## Contents

- [`greater()`](#jax.numpy.greater)

# jax.numpy.greater[\#](#jax-numpy-greater "Link to this heading")

jax.numpy.greater(*x*, *y*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L2027-L2072)[\#](#jax.numpy.greater "Link to this definition")  
Return element-wise truth value of `x`` ``>`` ``y`.

JAX implementation of [`numpy.greater`](https://numpy.org/doc/stable/reference/generated/numpy.greater.html#numpy.greater "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – input array or scalar.

- **y** (*ArrayLike*) – input array or scalar. `x` and `y` must either have same shape or be broadcast compatible.

Returns:  
An array containing boolean values. `True` if the elements of `x`` ``>`` ``y`, and `False` otherwise.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.less()`](jax.numpy.less.html#jax.numpy.less "jax.numpy.less"): Returns element-wise truth value of `x`` ``<`` ``y`.

- [`jax.numpy.greater_equal()`](jax.numpy.greater_equal.html#jax.numpy.greater_equal "jax.numpy.greater_equal"): Returns element-wise truth value of `x`` ``>=`` ``y`.

- [`jax.numpy.less_equal()`](jax.numpy.less_equal.html#jax.numpy.less_equal "jax.numpy.less_equal"): Returns element-wise truth value of `x`` ``<=`` ``y`.

Examples

Scalar inputs:

    >>> jnp.greater(5, 2)
    Array(True, dtype=bool, weak_type=True)

Inputs with same shape:

    >>> x = jnp.array([5, 9, -2])
    >>> y = jnp.array([4, -1, 6])
    >>> jnp.greater(x, y)
    Array([ True,  True, False], dtype=bool)

Inputs with broadcast compatibility:

    >>> x1 = jnp.array([[5, -6, 7],
    ...                 [-2, 5, 9]])
    >>> y1 = jnp.array([-4, 3, 10])
    >>> jnp.greater(x1, y1)
    Array([[ True, False, False],
           [ True,  True, False]], dtype=bool)

[](jax.numpy.gradient.html "previous page")

previous

jax.numpy.gradient

[](jax.numpy.greater_equal.html "next page")

next

jax.numpy.greater_equal

Contents

- [`greater()`](#jax.numpy.greater)

By The JAX authors

© Copyright 2024, The JAX Authors.\
