- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.greater_equal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.greater_equal.rst "Download source file")
-  .pdf

# jax.numpy.greater_equal

## Contents

- [`greater_equal()`](#jax.numpy.greater_equal)

# jax.numpy.greater_equal[\#](#jax-numpy-greater-equal "Link to this heading")

jax.numpy.greater_equal(*x*, *y*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L1981-L2025)[\#](#jax.numpy.greater_equal "Link to this definition")  
Return element-wise truth value of `x`` ``>=`` ``y`.

JAX implementation of [`numpy.greater_equal`](https://numpy.org/doc/stable/reference/generated/numpy.greater_equal.html#numpy.greater_equal "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – input array or scalar.

- **y** (*ArrayLike*) – input array or scalar. `x` and `y` must either have same shape or be broadcast compatible.

Returns:  
An array containing boolean values. `True` if the elements of `x`` ``>=`` ``y`, and `False` otherwise.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.less_equal()`](jax.numpy.less_equal.html#jax.numpy.less_equal "jax.numpy.less_equal"): Returns element-wise truth value of `x`` ``<=`` ``y`.

- [`jax.numpy.greater()`](jax.numpy.greater.html#jax.numpy.greater "jax.numpy.greater"): Returns element-wise truth value of `x`` ``>`` ``y`.

- [`jax.numpy.less()`](jax.numpy.less.html#jax.numpy.less "jax.numpy.less"): Returns element-wise truth value of `x`` ``<`` ``y`.

Examples

Scalar inputs:

    >>> jnp.greater_equal(4, 7)
    Array(False, dtype=bool, weak_type=True)

Inputs with same shape:

    >>> x = jnp.array([2, 5, -1])
    >>> y = jnp.array([-6, 4, 3])
    >>> jnp.greater_equal(x, y)
    Array([ True,  True, False], dtype=bool)

Inputs with broadcast compatibility:

    >>> x1 = jnp.array([[3, -1, 4],
    ...                 [5, 9, -6]])
    >>> y1 = jnp.array([-1, 4, 2])
    >>> jnp.greater_equal(x1, y1)
    Array([[ True, False,  True],
           [ True,  True, False]], dtype=bool)

[](jax.numpy.greater.html "previous page")

previous

jax.numpy.greater

[](jax.numpy.hamming.html "next page")

next

jax.numpy.hamming

Contents

- [`greater_equal()`](#jax.numpy.greater_equal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
