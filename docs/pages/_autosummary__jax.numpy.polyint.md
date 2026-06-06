- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.polyint

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.polyint.rst "Download source file")
-  .pdf

# jax.numpy.polyint

## Contents

- [`polyint()`](#jax.numpy.polyint)

# jax.numpy.polyint[\#](#jax-numpy-polyint "Link to this heading")

jax.numpy.polyint(*p*, *m=1*, *k=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/polynomial.py#L511-L578)[\#](#jax.numpy.polyint "Link to this definition")  
Returns the coefficients of the integration of specified order of a polynomial.

JAX implementation of [`numpy.polyint()`](https://numpy.org/doc/stable/reference/generated/numpy.polyint.html#numpy.polyint "(in NumPy v2.4)").

Parameters:  
- **p** (*ArrayLike*) – An array of polynomial coefficients.

- **m** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Order of integration. Default is 1. It must be specified statically.

- **k** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *ArrayLike* *\|* *None*) – Scalar or array of `m` integration constant (s).

Returns:  
An array of coefficients of integrated polynomial.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.polyder()`](jax.numpy.polyder.html#jax.numpy.polyder "jax.numpy.polyder"): Computes the coefficients of the derivative of a polynomial.

- [`jax.numpy.polyval()`](jax.numpy.polyval.html#jax.numpy.polyval "jax.numpy.polyval"): Evaluates a polynomial at specific values.

Examples

The first order integration of the polynomial \\12 x^2 + 12 x + 6\\ is \\4 x^3 + 6 x^2 + 6 x\\.

    >>> p = jnp.array([12, 12, 6])
    >>> jnp.polyint(p)
    Array([4., 6., 6., 0.], dtype=float32)

Since the constant `k` is not provided, the result included `0` at the end. If the constant `k` is provided:

    >>> jnp.polyint(p, k=4)
    Array([4., 6., 6., 4.], dtype=float32)

and the second order integration is \\x^4 + 2 x^3 + 3 x\\:

    >>> jnp.polyint(p, m=2)
    Array([1., 2., 3., 0., 0.], dtype=float32)

When `m>=2`, the constants `k` should be provided as an array having `m` elements. The second order integration of the polynomial \\12 x^2 + 12 x + 6\\ with the constants `k=[4,`` ``5]` is \\x^4 + 2 x^3 + 3 x^2 + 4 x + 5\\:

    >>> jnp.polyint(p, m=2, k=jnp.array([4, 5]))
    Array([1., 2., 3., 4., 5.], dtype=float32)

[](jax.numpy.polyfit.html "previous page")

previous

jax.numpy.polyfit

[](jax.numpy.polymul.html "next page")

next

jax.numpy.polymul

Contents

- [`polyint()`](#jax.numpy.polyint)

By The JAX authors

© Copyright 2024, The JAX Authors.\
