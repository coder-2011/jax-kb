- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.polymul

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.polymul.rst "Download source file")
-  .pdf

# jax.numpy.polymul

## Contents

- [`polymul()`](#jax.numpy.polymul)

# jax.numpy.polymul[\#](#jax-numpy-polymul "Link to this heading")

jax.numpy.polymul(*a1*, *a2*, *\**, *trim_leading_zeros=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/polynomial.py#L631-L696)[\#](#jax.numpy.polymul "Link to this definition")  
Returns the product of two polynomials.

JAX implementation of [`numpy.polymul()`](https://numpy.org/doc/stable/reference/generated/numpy.polymul.html#numpy.polymul "(in NumPy v2.4)").

Parameters:  
- **a1** (*ArrayLike*) – 1D array of polynomial coefficients.

- **a2** (*ArrayLike*) – 1D array of polynomial coefficients.

- **trim_leading_zeros** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Default is `False`. If `True` removes the leading zeros in the return value to match the result of numpy. But prevents the function from being able to be used in compiled code. Due to differences in accumulation of floating point arithmetic errors, the cutoff for values to be considered zero may lead to inconsistent results between NumPy and JAX, and even between different JAX backends. The result may lead to inconsistent output shapes when `trim_leading_zeros=True`.

Returns:  
An array of the coefficients of the product of the two polynomials. The dtype of the output is always promoted to inexact.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

[`jax.numpy.polymul()`](#jax.numpy.polymul "jax.numpy.polymul") only accepts arrays as input unlike [`numpy.polymul()`](https://numpy.org/doc/stable/reference/generated/numpy.polymul.html#numpy.polymul "(in NumPy v2.4)") which accepts scalar inputs as well.

See also

- [`jax.numpy.polyadd()`](jax.numpy.polyadd.html#jax.numpy.polyadd "jax.numpy.polyadd"): Computes the sum of two polynomials.

- [`jax.numpy.polysub()`](jax.numpy.polysub.html#jax.numpy.polysub "jax.numpy.polysub"): Computes the difference of two polynomials.

- [`jax.numpy.polydiv()`](jax.numpy.polydiv.html#jax.numpy.polydiv "jax.numpy.polydiv"): Computes the quotient and remainder of polynomial division.

Examples

    >>> x1 = np.array([2, 1, 0])
    >>> x2 = np.array([0, 5, 0, 3])
    >>> np.polymul(x1, x2)
    array([10,  5,  6,  3,  0])
    >>> jnp.polymul(x1, x2)
    Array([ 0., 10.,  5.,  6.,  3.,  0.], dtype=float32)

If `trim_leading_zeros=True`, the result matches with `np.polymul`’s.

    >>> jnp.polymul(x1, x2, trim_leading_zeros=True)
    Array([10.,  5.,  6.,  3.,  0.], dtype=float32)

For input arrays of dtype `complex`:

    >>> x3 = np.array([2., 1+2j, 1-2j])
    >>> x4 = np.array([0, 5, 0, 3])
    >>> np.polymul(x3, x4)
    array([10. +0.j,  5.+10.j, 11.-10.j,  3. +6.j,  3. -6.j])
    >>> jnp.polymul(x3, x4)
    Array([ 0. +0.j, 10. +0.j,  5.+10.j, 11.-10.j,  3. +6.j,  3. -6.j],      dtype=complex64)
    >>> jnp.polymul(x3, x4, trim_leading_zeros=True)
    Array([10. +0.j,  5.+10.j, 11.-10.j,  3. +6.j,  3. -6.j], dtype=complex64)

[](jax.numpy.polyint.html "previous page")

previous

jax.numpy.polyint

[](jax.numpy.polysub.html "next page")

next

jax.numpy.polysub

Contents

- [`polymul()`](#jax.numpy.polymul)

By The JAX authors

© Copyright 2024, The JAX Authors.\
