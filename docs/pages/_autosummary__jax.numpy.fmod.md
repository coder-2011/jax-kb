- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fmod

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fmod.rst "Download source file")
-  .pdf

# jax.numpy.fmod

## Contents

- [`fmod()`](#jax.numpy.fmod)

# jax.numpy.fmod[\#](#jax-numpy-fmod "Link to this heading")

jax.numpy.fmod(*x1*, *x2*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3149-L3191)[\#](#jax.numpy.fmod "Link to this definition")  
Calculate element-wise floating-point modulo operation.

JAX implementation of [`numpy.fmod`](https://numpy.org/doc/stable/reference/generated/numpy.fmod.html#numpy.fmod "(in NumPy v2.4)").

Parameters:  
- **x1** (*ArrayLike*) – scalar or array. Specifies the dividend.

- **x2** (*ArrayLike*) – scalar or array. Specifies the divisor. `x1` and `x2` should either have same shape or be broadcast compatible.

Returns:  
An array containing the result of the element-wise floating-point modulo operation of `x1` and `x2` with same sign as the elements of `x1`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

The result of `jnp.fmod` is equivalent to `x1`` ``-`` ``x2`` ``*`` ``jnp.trunc(x1`` ``/`` ``x2)`.

See also

- [`jax.numpy.mod()`](jax.numpy.mod.html#jax.numpy.mod "jax.numpy.mod") and [`jax.numpy.remainder()`](jax.numpy.remainder.html#jax.numpy.remainder "jax.numpy.remainder"): Returns the element-wise remainder of the division.

- [`jax.numpy.divmod()`](jax.numpy.divmod.html#jax.numpy.divmod "jax.numpy.divmod"): Calculates the integer quotient and remainder of `x1` by `x2`, element-wise.

Examples

    >>> x1 = jnp.array([[3, -1, 4],
    ...                 [8, 5, -2]])
    >>> x2 = jnp.array([2, 3, -5])
    >>> jnp.fmod(x1, x2)
    Array([[ 1, -1,  4],
           [ 0,  2, -2]], dtype=int32)
    >>> x1 - x2 * jnp.trunc(x1 / x2)
    Array([[ 1., -1.,  4.],
           [ 0.,  2., -2.]], dtype=float32)

[](jax.numpy.fmin.html "previous page")

previous

jax.numpy.fmin

[](jax.numpy.frexp.html "next page")

next

jax.numpy.frexp

Contents

- [`fmod()`](#jax.numpy.fmod)

By The JAX authors

© Copyright 2024, The JAX Authors.\
