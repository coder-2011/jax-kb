- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.polydiv

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.polydiv.rst "Download source file")
-  .pdf

# jax.numpy.polydiv

## Contents

- [`polydiv()`](#jax.numpy.polydiv)

# jax.numpy.polydiv[\#](#jax-numpy-polydiv "Link to this heading")

jax.numpy.polydiv(*u*, *v*, *\**, *trim_leading_zeros=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/polynomial.py#L698-L756)[\#](#jax.numpy.polydiv "Link to this definition")  
Returns the quotient and remainder of polynomial division.

JAX implementation of [`numpy.polydiv()`](https://numpy.org/doc/stable/reference/generated/numpy.polydiv.html#numpy.polydiv "(in NumPy v2.4)").

Parameters:  
- **u** (*ArrayLike*) – Array of dividend polynomial coefficients.

- **v** (*ArrayLike*) – Array of divisor polynomial coefficients.

- **trim_leading_zeros** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Default is `False`. If `True` removes the leading zeros in the return value to match the result of numpy. But prevents the function from being able to be used in compiled code. Due to differences in accumulation of floating point arithmetic errors, the cutoff for values to be considered zero may lead to inconsistent results between NumPy and JAX, and even between different JAX backends. The result may lead to inconsistent output shapes when `trim_leading_zeros=True`.

Returns:  
A tuple of quotient and remainder arrays. The dtype of the output is always promoted to inexact.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

Note

[`jax.numpy.polydiv()`](#jax.numpy.polydiv "jax.numpy.polydiv") only accepts arrays as input unlike [`numpy.polydiv()`](https://numpy.org/doc/stable/reference/generated/numpy.polydiv.html#numpy.polydiv "(in NumPy v2.4)") which accepts scalar inputs as well.

See also

- [`jax.numpy.polyadd()`](jax.numpy.polyadd.html#jax.numpy.polyadd "jax.numpy.polyadd"): Computes the sum of two polynomials.

- [`jax.numpy.polysub()`](jax.numpy.polysub.html#jax.numpy.polysub "jax.numpy.polysub"): Computes the difference of two polynomials.

- [`jax.numpy.polymul()`](jax.numpy.polymul.html#jax.numpy.polymul "jax.numpy.polymul"): Computes the product of two polynomials.

Examples

    >>> x1 = jnp.array([5, 7, 9])
    >>> x2 = jnp.array([4, 1])
    >>> np.polydiv(x1, x2)
    (array([1.25  , 1.4375]), array([7.5625]))
    >>> jnp.polydiv(x1, x2)
    (Array([1.25  , 1.4375], dtype=float32), Array([0.    , 0.    , 7.5625], dtype=float32))

If `trim_leading_zeros=True`, the result matches with `np.polydiv`’s.

    >>> jnp.polydiv(x1, x2, trim_leading_zeros=True)
    (Array([1.25  , 1.4375], dtype=float32), Array([7.5625], dtype=float32))

[](jax.numpy.polyder.html "previous page")

previous

jax.numpy.polyder

[](jax.numpy.polyfit.html "next page")

next

jax.numpy.polyfit

Contents

- [`polydiv()`](#jax.numpy.polydiv)

By The JAX authors

© Copyright 2024, The JAX Authors.\
