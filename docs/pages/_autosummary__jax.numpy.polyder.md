- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.polyder

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.polyder.rst "Download source file")
-  .pdf

# jax.numpy.polyder

## Contents

- [`polyder()`](#jax.numpy.polyder)

# jax.numpy.polyder[\#](#jax-numpy-polyder "Link to this heading")

jax.numpy.polyder(*p*, *m=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/polynomial.py#L580-L629)[\#](#jax.numpy.polyder "Link to this definition")  
Returns the coefficients of the derivative of specified order of a polynomial.

JAX implementation of [`numpy.polyder()`](https://numpy.org/doc/stable/reference/generated/numpy.polyder.html#numpy.polyder "(in NumPy v2.4)").

Parameters:  
- **p** (*ArrayLike*) – Array of polynomials coefficients.

- **m** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Order of differentiation (positive integer). Default is 1. It must be specified statically.

Returns:  
An array of polynomial coefficients representing the derivative.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

[`jax.numpy.polyder()`](#jax.numpy.polyder "jax.numpy.polyder") differs from [`numpy.polyder()`](https://numpy.org/doc/stable/reference/generated/numpy.polyder.html#numpy.polyder "(in NumPy v2.4)") when an integer array is given. NumPy returns the result with dtype `int` whereas JAX returns the result with dtype `float`.

See also

- [`jax.numpy.polyint()`](jax.numpy.polyint.html#jax.numpy.polyint "jax.numpy.polyint"): Computes the integral of polynomial.

- [`jax.numpy.polyval()`](jax.numpy.polyval.html#jax.numpy.polyval "jax.numpy.polyval"): Evaluates a polynomial at specific values.

Examples

The first order derivative of the polynomial \\2 x^3 - 5 x^2 + 3 x - 1\\ is \\6 x^2 - 10 x +3\\:

    >>> p = jnp.array([2, -5, 3, -1])
    >>> jnp.polyder(p)
    Array([  6., -10.,   3.], dtype=float32)

and its second order derivative is \\12 x - 10\\:

    >>> jnp.polyder(p, m=2)
    Array([ 12., -10.], dtype=float32)

[](jax.numpy.polyadd.html "previous page")

previous

jax.numpy.polyadd

[](jax.numpy.polydiv.html "next page")

next

jax.numpy.polydiv

Contents

- [`polyder()`](#jax.numpy.polyder)

By The JAX authors

© Copyright 2024, The JAX Authors.\
