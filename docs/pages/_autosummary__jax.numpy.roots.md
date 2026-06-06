- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.roots

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.roots.rst "Download source file")
-  .pdf

# jax.numpy.roots

## Contents

- [`roots()`](#jax.numpy.roots)

# jax.numpy.roots[\#](#jax-numpy-roots "Link to this heading")

jax.numpy.roots(*p*, *\**, *strip_zeros=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/polynomial.py#L65-L124)[\#](#jax.numpy.roots "Link to this definition")  
Returns the roots of a polynomial given the coefficients `p`.

JAX implementations of [`numpy.roots()`](https://numpy.org/doc/stable/reference/generated/numpy.roots.html#numpy.roots "(in NumPy v2.4)").

Parameters:  
- **p** (*ArrayLike*) – Array of polynomial coefficients having rank-1.

- **strip_zeros** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=True. If True, then leading zeros in the coefficients will be stripped, similar to [`numpy.roots()`](https://numpy.org/doc/stable/reference/generated/numpy.roots.html#numpy.roots "(in NumPy v2.4)"). If set to False, leading zeros will not be stripped, and undefined roots will be represented by NaN values in the function output. `strip_zeros` must be set to `False` for the function to be compatible with [`jax.jit()`](jax.jit.html#jax.jit "jax.jit") and other JAX transformations.

Returns:  
An array containing the roots of the polynomial.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

Unlike `np.roots` of this function, the `jnp.roots` returns the roots in a complex array regardless of the values of the roots.

See also

- [`jax.numpy.poly()`](jax.numpy.poly.html#jax.numpy.poly "jax.numpy.poly"): Finds the polynomial coefficients of the given sequence of roots.

- [`jax.numpy.polyfit()`](jax.numpy.polyfit.html#jax.numpy.polyfit "jax.numpy.polyfit"): Least squares polynomial fit to data.

- [`jax.numpy.polyval()`](jax.numpy.polyval.html#jax.numpy.polyval "jax.numpy.polyval"): Evaluate a polynomial at specific values.

Examples

    >>> coeffs = jnp.array([0, 1, 2])

The default behavior matches numpy and strips leading zeros:

    >>> jnp.roots(coeffs)
    Array([-2.+0.j], dtype=complex64)

With `strip_zeros=False`, extra roots are set to NaN:

    >>> jnp.roots(coeffs, strip_zeros=False)
    Array([-2. +0.j, nan+nanj], dtype=complex64)

[](jax.numpy.rollaxis.html "previous page")

previous

jax.numpy.rollaxis

[](jax.numpy.rot90.html "next page")

next

jax.numpy.rot90

Contents

- [`roots()`](#jax.numpy.roots)

By The JAX authors

© Copyright 2024, The JAX Authors.\
