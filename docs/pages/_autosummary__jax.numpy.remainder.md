- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.remainder

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.remainder.rst "Download source file")
-  .pdf

# jax.numpy.remainder

## Contents

- [`remainder()`](#jax.numpy.remainder)

# jax.numpy.remainder[\#](#jax-numpy-remainder "Link to this heading")

jax.numpy.remainder(*x1*, *x2*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3092-L3141)[\#](#jax.numpy.remainder "Link to this definition")  
Returns element-wise remainder of the division.

JAX implementation of [`numpy.remainder`](https://numpy.org/doc/stable/reference/generated/numpy.remainder.html#numpy.remainder "(in NumPy v2.4)").

Parameters:  
- **x1** (*ArrayLike*) – scalar or array. Specifies the dividend.

- **x2** (*ArrayLike*) – scalar or array. Specifies the divisor. `x1` and `x2` should either have same shape or be broadcast compatible.

Returns:  
An array containing the remainder of element-wise division of `x1` by `x2` with same sign as the elements of `x2`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

The result of `jnp.remainder` is equivalent to `x1`` ``-`` ``x2`` ``*`` ``jnp.floor(x1`` ``/`` ``x2)`.

See also

- [`jax.numpy.mod()`](jax.numpy.mod.html#jax.numpy.mod "jax.numpy.mod"): Returns the element-wise remainder of the division.

- [`jax.numpy.fmod()`](jax.numpy.fmod.html#jax.numpy.fmod "jax.numpy.fmod"): Calculates the element-wise floating-point modulo operation.

- [`jax.numpy.divmod()`](jax.numpy.divmod.html#jax.numpy.divmod "jax.numpy.divmod"): Calculates the integer quotient and remainder of `x1` by `x2`, element-wise.

Examples

    >>> x1 = jnp.array([[3, -1, 4],
    ...                 [8, 5, -2]])
    >>> x2 = jnp.array([2, 3, -5])
    >>> jnp.remainder(x1, x2)
    Array([[ 1,  2, -1],
           [ 0,  2, -2]], dtype=int32)
    >>> x1 - x2 * jnp.floor(x1 / x2)
    Array([[ 1.,  2., -1.],
           [ 0.,  2., -2.]], dtype=float32)

[](jax.numpy.reciprocal.html "previous page")

previous

jax.numpy.reciprocal

[](jax.numpy.repeat.html "next page")

next

jax.numpy.repeat

Contents

- [`remainder()`](#jax.numpy.remainder)

By The JAX authors

© Copyright 2024, The JAX Authors.\
