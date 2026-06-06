- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.polyval

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.polyval.rst "Download source file")
-  .pdf

# jax.numpy.polyval

## Contents

- [`polyval()`](#jax.numpy.polyval)

# jax.numpy.polyval[\#](#jax-numpy-polyval "Link to this heading")

jax.numpy.polyval(*p*, *x*, *\**, *unroll=16*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/polynomial.py#L389-L451)[\#](#jax.numpy.polyval "Link to this definition")  
Evaluates the polynomial at specific values.

JAX implementations of [`numpy.polyval()`](https://numpy.org/doc/stable/reference/generated/numpy.polyval.html#numpy.polyval "(in NumPy v2.4)").

For the 1D-polynomial coefficients `p` of length `M`, the function returns the value:

\\p_0 x^{M - 1} + p_1 x^{M - 2} + ... + p\_{M - 1}\\

Parameters:  
- **p** (*ArrayLike*) – An array of polynomial coefficients of shape `(M,)`.

- **x** (*ArrayLike*) – A number or an array of numbers.

- **unroll** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – A number used to control the number of unrolled steps with `lax.scan`. It must be specified statically.

Returns:  
An array of same shape as `x`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

The `unroll` parameter is JAX specific. It does not affect correctness but can have a major impact on performance for evaluating high-order polynomials. The parameter controls the number of unrolled steps with `lax.scan` inside the `jnp.polyval` implementation. Consider setting `unroll=128` (or even higher) to improve runtime performance on accelerators, at the cost of increased compilation time.

See also

- [`jax.numpy.polyfit()`](jax.numpy.polyfit.html#jax.numpy.polyfit "jax.numpy.polyfit"): Least squares polynomial fit.

- [`jax.numpy.poly()`](jax.numpy.poly.html#jax.numpy.poly "jax.numpy.poly"): Finds the coefficients of a polynomial with given roots.

- [`jax.numpy.roots()`](jax.numpy.roots.html#jax.numpy.roots "jax.numpy.roots"): Computes the roots of a polynomial for given coefficients.

Examples

    >>> p = jnp.array([2, 5, 1])
    >>> jnp.polyval(p, 3)
    Array(34, dtype=int32)

If `x` is a 2D array, `polyval` returns 2D-array with same shape as that of `x`:

    >>> x = jnp.array([[2, 1, 5],
    ...                [3, 4, 7],
    ...                [1, 3, 5]])
    >>> jnp.polyval(p, x)
    Array([[ 19,   8,  76],
           [ 34,  53, 134],
           [  8,  34,  76]], dtype=int32)

[](jax.numpy.polysub.html "previous page")

previous

jax.numpy.polysub

[](jax.numpy.positive.html "next page")

next

jax.numpy.positive

Contents

- [`polyval()`](#jax.numpy.polyval)

By The JAX authors

© Copyright 2024, The JAX Authors.\
