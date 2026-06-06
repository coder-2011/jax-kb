- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.rad2deg

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.rad2deg.rst "Download source file")
-  .pdf

# jax.numpy.rad2deg

## Contents

- [`rad2deg()`](#jax.numpy.rad2deg)

# jax.numpy.rad2deg[\#](#jax-numpy-rad2deg "Link to this heading")

jax.numpy.rad2deg(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3278-L3312)[\#](#jax.numpy.rad2deg "Link to this definition")  
Convert angles from radians to degrees.

JAX implementation of [`numpy.rad2deg`](https://numpy.org/doc/stable/reference/generated/numpy.rad2deg.html#numpy.rad2deg "(in NumPy v2.4)").

The angle in radians is converted to degrees by:

\\rad2deg(x) = x \* \frac{180}{pi}\\

Parameters:  
**x** (*ArrayLike*) – scalar or array. Specifies the angle in radians.

Returns:  
An array containing the angles in degrees.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.deg2rad()`](jax.numpy.deg2rad.html#jax.numpy.deg2rad "jax.numpy.deg2rad") and [`jax.numpy.radians()`](jax.numpy.radians.html#jax.numpy.radians "jax.numpy.radians"): Converts the angles from degrees to radians.

- [`jax.numpy.degrees()`](jax.numpy.degrees.html#jax.numpy.degrees "jax.numpy.degrees"): Alias of `rad2deg`.

Examples

    >>> pi = jnp.pi
    >>> x = jnp.array([pi/4, pi/2, 2*pi/3])
    >>> jnp.rad2deg(x)
    Array([ 45.     ,  90.     , 120.00001], dtype=float32)
    >>> x * 180 / pi
    Array([ 45.     ,  90.     , 119.99999], dtype=float32)

[](jax.numpy.r_.html "previous page")

previous

jax.numpy.r\_

[](jax.numpy.radians.html "next page")

next

jax.numpy.radians

Contents

- [`rad2deg()`](#jax.numpy.rad2deg)

By The JAX authors

© Copyright 2024, The JAX Authors.\
