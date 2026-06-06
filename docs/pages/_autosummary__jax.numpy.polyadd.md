- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.polyadd

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.polyadd.rst "Download source file")
-  .pdf

# jax.numpy.polyadd

## Contents

- [`polyadd()`](#jax.numpy.polyadd)

# jax.numpy.polyadd[\#](#jax-numpy-polyadd "Link to this heading")

jax.numpy.polyadd(*a1*, *a2*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/polynomial.py#L453-L509)[\#](#jax.numpy.polyadd "Link to this definition")  
Returns the sum of the two polynomials.

JAX implementation of [`numpy.polyadd()`](https://numpy.org/doc/stable/reference/generated/numpy.polyadd.html#numpy.polyadd "(in NumPy v2.4)").

Parameters:  
- **a1** (*ArrayLike*) – Array of polynomial coefficients.

- **a2** (*ArrayLike*) – Array of polynomial coefficients.

Returns:  
An array containing the coefficients of the sum of input polynomials.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

[`jax.numpy.polyadd()`](#jax.numpy.polyadd "jax.numpy.polyadd") only accepts arrays as input unlike [`numpy.polyadd()`](https://numpy.org/doc/stable/reference/generated/numpy.polyadd.html#numpy.polyadd "(in NumPy v2.4)") which accepts scalar inputs as well.

See also

- [`jax.numpy.polysub()`](jax.numpy.polysub.html#jax.numpy.polysub "jax.numpy.polysub"): Computes the difference of two polynomials.

- [`jax.numpy.polymul()`](jax.numpy.polymul.html#jax.numpy.polymul "jax.numpy.polymul"): Computes the product of two polynomials.

- [`jax.numpy.polydiv()`](jax.numpy.polydiv.html#jax.numpy.polydiv "jax.numpy.polydiv"): Computes the quotient and remainder of polynomial division.

Examples

    >>> x1 = jnp.array([2, 3])
    >>> x2 = jnp.array([5, 4, 1])
    >>> jnp.polyadd(x1, x2)
    Array([5, 6, 4], dtype=int32)

    >>> x3 = jnp.array([[2, 3, 1]])
    >>> x4 = jnp.array([[5, 7, 3],
    ...                 [8, 2, 6]])
    >>> jnp.polyadd(x3, x4)
    Array([[ 5,  7,  3],
           [10,  5,  7]], dtype=int32)

    >>> x5 = jnp.array([1, 3, 5])
    >>> x6 = jnp.array([[5, 7, 9],
    ...                 [8, 6, 4]])
    >>> jnp.polyadd(x5, x6)  
    Traceback (most recent call last):
    ...
    ValueError: Cannot broadcast to shape with fewer dimensions: arr_shape=(2, 3) shape=(2,)
    >>> x7 = jnp.array([2])
    >>> jnp.polyadd(x6, x7)
    Array([[ 5,  7,  9],
           [10,  8,  6]], dtype=int32)

[](jax.numpy.poly.html "previous page")

previous

jax.numpy.poly

[](jax.numpy.polyder.html "next page")

next

jax.numpy.polyder

Contents

- [`polyadd()`](#jax.numpy.polyadd)

By The JAX authors

© Copyright 2024, The JAX Authors.\
