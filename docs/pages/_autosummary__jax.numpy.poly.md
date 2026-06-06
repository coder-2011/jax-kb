- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.poly

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.poly.rst "Download source file")
-  .pdf

# jax.numpy.poly

## Contents

- [`poly()`](#jax.numpy.poly)

# jax.numpy.poly[\#](#jax-numpy-poly "Link to this heading")

jax.numpy.poly(*seq_of_zeros*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/polynomial.py#L306-L387)[\#](#jax.numpy.poly "Link to this definition")  
Returns the coefficients of a polynomial for the given sequence of roots.

JAX implementation of [`numpy.poly()`](https://numpy.org/doc/stable/reference/generated/numpy.poly.html#numpy.poly "(in NumPy v2.4)").

Parameters:  
**seq_of_zeros** (*ArrayLike*) – A scalar or an array of roots of the polynomial of shape `(M,)` or `(M,`` ``M)`.

Returns:  
An array containing the coefficients of the polynomial. The dtype of the output is always promoted to inexact.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

[`jax.numpy.poly()`](#jax.numpy.poly "jax.numpy.poly") differs from [`numpy.poly()`](https://numpy.org/doc/stable/reference/generated/numpy.poly.html#numpy.poly "(in NumPy v2.4)"):

- When the input is a scalar, `np.poly` raises a `TypeError`, whereas `jnp.poly` treats scalars the same as length-1 arrays.

- For complex-valued or square-shaped inputs, `jnp.poly` always returns complex coefficients, whereas `np.poly` may return real or complex depending on their values.

See also

- [`jax.numpy.polyfit()`](jax.numpy.polyfit.html#jax.numpy.polyfit "jax.numpy.polyfit"): Least squares polynomial fit.

- [`jax.numpy.polyval()`](jax.numpy.polyval.html#jax.numpy.polyval "jax.numpy.polyval"): Evaluate a polynomial at specific values.

- [`jax.numpy.roots()`](jax.numpy.roots.html#jax.numpy.roots "jax.numpy.roots"): Computes the roots of a polynomial for given coefficients.

Examples

Scalar inputs:

    >>> jnp.poly(1)
    Array([ 1., -1.], dtype=float32)

Input array with integer values:

    >>> x = jnp.array([1, 2, 3])
    >>> jnp.poly(x)
    Array([ 1., -6., 11., -6.], dtype=float32)

Input array with complex conjugates:

    >>> x = jnp.array([2, 1+2j, 1-2j])
    >>> jnp.poly(x)
    Array([  1.+0.j,  -4.+0.j,   9.+0.j, -10.+0.j], dtype=complex64)

Input array as square matrix with real valued inputs:

    >>> x = jnp.array([[2, 1, 5],
    ...                [3, 4, 7],
    ...                [1, 3, 5]])
    >>> jnp.round(jnp.poly(x))
    Array([  1.+0.j, -11.-0.j,   9.+0.j, -15.+0.j], dtype=complex64)

[](jax.numpy.place.html "previous page")

previous

jax.numpy.place

[](jax.numpy.polyadd.html "next page")

next

jax.numpy.polyadd

Contents

- [`poly()`](#jax.numpy.poly)

By The JAX authors

© Copyright 2024, The JAX Authors.\
