- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.angle

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.angle.rst "Download source file")
-  .pdf

# jax.numpy.angle

## Contents

- [`angle()`](#jax.numpy.angle)

# jax.numpy.angle[\#](#jax-numpy-angle "Link to this heading")

jax.numpy.angle(*z*, *deg=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L1507-L1558)[\#](#jax.numpy.angle "Link to this definition")  
Return the angle of a complex valued number or array.

JAX implementation of [`numpy.angle()`](https://numpy.org/doc/stable/reference/generated/numpy.angle.html#numpy.angle "(in NumPy v2.4)").

Parameters:  
- **z** (*ArrayLike*) – A complex number or an array of complex numbers.

- **deg** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Boolean. If `True`, returns the result in degrees else returns in radians. Default is `False`.

Returns:  
An array of counterclockwise angle of each element of `z`, with the same shape as `z` of dtype float.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

If `z` is a number

    >>> z1 = 2+3j
    >>> jnp.angle(z1)
    Array(0.98279375, dtype=float32, weak_type=True)

If `z` is an array

    >>> z2 = jnp.array([[1+3j, 2-5j],
    ...                 [4-3j, 3+2j]])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...     print(jnp.angle(z2))
    [[ 1.25 -1.19]
     [-0.64  0.59]]

If `deg=True`.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...     print(jnp.angle(z2, deg=True))
    [[ 71.57 -68.2 ]
     [-36.87  33.69]]

[](jax.numpy.amin.html "previous page")

previous

jax.numpy.amin

[](jax.numpy.any.html "next page")

next

jax.numpy.any

Contents

- [`angle()`](#jax.numpy.angle)

By The JAX authors

© Copyright 2024, The JAX Authors.\
