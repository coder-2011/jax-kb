- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.deg2rad

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.deg2rad.rst "Download source file")
-  .pdf

# jax.numpy.deg2rad

## Contents

- [`deg2rad()`](#jax.numpy.deg2rad)

# jax.numpy.deg2rad[\#](#jax-numpy-deg2rad "Link to this heading")

jax.numpy.deg2rad(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3243-L3276)[\#](#jax.numpy.deg2rad "Link to this definition")  
Convert angles from degrees to radians.

JAX implementation of [`numpy.deg2rad`](https://numpy.org/doc/stable/reference/generated/numpy.deg2rad.html#numpy.deg2rad "(in NumPy v2.4)").

The angle in degrees is converted to radians by:

\\deg2rad(x) = x \* \frac{pi}{180}\\

Parameters:  
**x** (*ArrayLike*) – scalar or array. Specifies the angle in degrees.

Returns:  
An array containing the angles in radians.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.rad2deg()`](jax.numpy.rad2deg.html#jax.numpy.rad2deg "jax.numpy.rad2deg") and [`jax.numpy.degrees()`](jax.numpy.degrees.html#jax.numpy.degrees "jax.numpy.degrees"): Converts the angles from radians to degrees.

- [`jax.numpy.radians()`](jax.numpy.radians.html#jax.numpy.radians "jax.numpy.radians"): Alias of `deg2rad`.

Examples

    >>> x = jnp.array([60, 90, 120, 180])
    >>> jnp.deg2rad(x)
    Array([1.0471976, 1.5707964, 2.0943952, 3.1415927], dtype=float32)
    >>> x * jnp.pi / 180
    Array([1.0471976, 1.5707964, 2.0943952, 3.1415927],      dtype=float32, weak_type=True)

[](jax.numpy.cumulative_sum.html "previous page")

previous

jax.numpy.cumulative_sum

[](jax.numpy.degrees.html "next page")

next

jax.numpy.degrees

Contents

- [`deg2rad()`](#jax.numpy.deg2rad)

By The JAX authors

© Copyright 2024, The JAX Authors.\
