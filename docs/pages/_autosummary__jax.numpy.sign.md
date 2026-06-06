- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.sign

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.sign.rst "Download source file")
-  .pdf

# jax.numpy.sign

## Contents

- [`sign()`](#jax.numpy.sign)

# jax.numpy.sign[\#](#jax-numpy-sign "Link to this heading")

jax.numpy.sign(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L285-L334)[\#](#jax.numpy.sign "Link to this definition")  
Return an element-wise indication of sign of the input.

JAX implementation of [`numpy.sign`](https://numpy.org/doc/stable/reference/generated/numpy.sign.html#numpy.sign "(in NumPy v2.4)").

The sign of `x` for real-valued input is:

\\\begin{split}\mathrm{sign}(x) = \begin{cases} 1, & x \> 0\\ 0, & x = 0\\ -1, & x \< 0 \end{cases}\end{split}\\

For complex valued input, `jnp.sign` returns a unit vector representing the phase. For generalized case, the sign of `x` is given by:

\\\begin{split}\mathrm{sign}(x) = \begin{cases} \frac{x}{abs(x)}, & x \ne 0\\ 0, & x = 0 \end{cases}\end{split}\\

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
An array with same shape and dtype as `x` containing the sign indication.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.positive()`](jax.numpy.positive.html#jax.numpy.positive "jax.numpy.positive"): Returns element-wise positive values of the input.

- [`jax.numpy.negative()`](jax.numpy.negative.html#jax.numpy.negative "jax.numpy.negative"): Returns element-wise negative values of the input.

Examples

For Real-valued inputs:

    >>> x = jnp.array([0., -3., 7.])
    >>> jnp.sign(x)
    Array([ 0., -1.,  1.], dtype=float32)

For complex-inputs:

    >>> x1 = jnp.array([1, 3+4j, 5j])
    >>> jnp.sign(x1)
    Array([1. +0.j , 0.6+0.8j, 0. +1.j ], dtype=complex64)

[](jax.numpy.shape.html "previous page")

previous

jax.numpy.shape

[](jax.numpy.signbit.html "next page")

next

jax.numpy.signbit

Contents

- [`sign()`](#jax.numpy.sign)

By The JAX authors

© Copyright 2024, The JAX Authors.\
