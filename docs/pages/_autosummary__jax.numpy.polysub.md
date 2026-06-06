- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.polysub

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.polysub.rst "Download source file")
-  .pdf

# jax.numpy.polysub

## Contents

- [`polysub()`](#jax.numpy.polysub)

# jax.numpy.polysub[\#](#jax-numpy-polysub "Link to this heading")

jax.numpy.polysub(*a1*, *a2*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/polynomial.py#L758-L810)[\#](#jax.numpy.polysub "Link to this definition")  
Returns the difference of two polynomials.

JAX implementation of [`numpy.polysub()`](https://numpy.org/doc/stable/reference/generated/numpy.polysub.html#numpy.polysub "(in NumPy v2.4)").

Parameters:  
- **a1** (*ArrayLike*) – Array of minuend polynomial coefficients.

- **a2** (*ArrayLike*) – Array of subtrahend polynomial coefficients.

Returns:  
An array containing the coefficients of the difference of two polynomials.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

[`jax.numpy.polysub()`](#jax.numpy.polysub "jax.numpy.polysub") only accepts arrays as input unlike [`numpy.polysub()`](https://numpy.org/doc/stable/reference/generated/numpy.polysub.html#numpy.polysub "(in NumPy v2.4)") which accepts scalar inputs as well.

See also

- [`jax.numpy.polyadd()`](jax.numpy.polyadd.html#jax.numpy.polyadd "jax.numpy.polyadd"): Computes the sum of two polynomials.

- [`jax.numpy.polymul()`](jax.numpy.polymul.html#jax.numpy.polymul "jax.numpy.polymul"): Computes the product of two polynomials.

- [`jax.numpy.polydiv()`](jax.numpy.polydiv.html#jax.numpy.polydiv "jax.numpy.polydiv"): Computes the quotient and remainder of polynomial division.

Examples

    >>> x1 = jnp.array([2, 3])
    >>> x2 = jnp.array([5, 4, 1])
    >>> jnp.polysub(x1, x2)
    Array([-5, -2,  2], dtype=int32)

    >>> x3 = jnp.array([[2, 3, 1]])
    >>> x4 = jnp.array([[5, 7, 3],
    ...                 [8, 2, 6]])
    >>> jnp.polysub(x3, x4)
    Array([[-5, -7, -3],
           [-6,  1, -5]], dtype=int32)

    >>> x5 = jnp.array([1, 3, 5])
    >>> x6 = jnp.array([[5, 7, 9],
    ...                 [8, 6, 4]])
    >>> jnp.polysub(x5, x6)  
    Traceback (most recent call last):
    ...
    ValueError: Cannot broadcast to shape with fewer dimensions: arr_shape=(2, 3) shape=(2,)
    >>> x7 = jnp.array([2])
    >>> jnp.polysub(x6, x7)
    Array([[5, 7, 9],
           [6, 4, 2]], dtype=int32)

[](jax.numpy.polymul.html "previous page")

previous

jax.numpy.polymul

[](jax.numpy.polyval.html "next page")

next

jax.numpy.polyval

Contents

- [`polysub()`](#jax.numpy.polysub)

By The JAX authors

© Copyright 2024, The JAX Authors.\
