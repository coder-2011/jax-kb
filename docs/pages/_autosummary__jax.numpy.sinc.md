- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.sinc

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.sinc.rst "Download source file")
-  .pdf

# jax.numpy.sinc

## Contents

- [`sinc()`](#jax.numpy.sinc)

# jax.numpy.sinc[\#](#jax-numpy-sinc "Link to this heading")

jax.numpy.sinc(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3796-L3851)[\#](#jax.numpy.sinc "Link to this definition")  
Calculate the normalized sinc function.

JAX implementation of [`numpy.sinc()`](https://numpy.org/doc/stable/reference/generated/numpy.sinc.html#numpy.sinc "(in NumPy v2.4)").

The normalized sinc function is given by

\\\mathrm{sinc}(x) = \frac{\sin({\pi x})}{\pi x}\\

where `sinc(0)` returns the limit value of `1`. The sinc function is smooth and infinitely differentiable.

Parameters:  
**x** (*ArrayLike*) – input array; will be promoted to an inexact type.

Returns:  
An array of the same shape as `x` containing the result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> x = jnp.array([-1, -0.5, 0, 0.5, 1])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.sinc(x)
    Array([-0.   ,  0.637,  1.   ,  0.637, -0.   ], dtype=float32)

Compare this to the naive approach to computing the function, which is undefined at zero:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.sin(jnp.pi * x) / (jnp.pi * x)
    Array([-0.   ,  0.637,    nan,  0.637, -0.   ], dtype=float32)

JAX defines a custom gradient rule for sinc to allow accurate evaluation of the gradient at zero even for higher-order derivatives:

    >>> f = jnp.sinc
    >>> for i in range(1, 6):
    ...   f = jax.grad(f)
    ...   print(f"(d/dx)^{i} f(0.0) = {f(0.0):.2f}")
    ...
    (d/dx)^1 f(0.0) = 0.00
    (d/dx)^2 f(0.0) = -3.29
    (d/dx)^3 f(0.0) = 0.00
    (d/dx)^4 f(0.0) = 19.48
    (d/dx)^5 f(0.0) = 0.00

[](jax.numpy.sin.html "previous page")

previous

jax.numpy.sin

[](jax.numpy.single.html "next page")

next

jax.numpy.single

Contents

- [`sinc()`](#jax.numpy.sinc)

By The JAX authors

© Copyright 2024, The JAX Authors.\
