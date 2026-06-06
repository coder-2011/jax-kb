- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.arctan2

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.arctan2.rst "Download source file")
-  .pdf

# jax.numpy.arctan2

## Contents

- [`arctan2()`](#jax.numpy.arctan2)

# jax.numpy.arctan2[\#](#jax-numpy-arctan2 "Link to this heading")

jax.numpy.arctan2(*x1*, *x2*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L1574-L1628)[\#](#jax.numpy.arctan2 "Link to this definition")  
Compute the arctangent of x1/x2, choosing the correct quadrant.

JAX implementation of `numpy.arctan2()`

Parameters:  
- **x1** (*ArrayLike*) – numerator array.

- **x2** (*ArrayLike*) – denomniator array; should be broadcast-compatible with x1.

Returns:  
The elementwise arctangent of x1 / x2, tracking the correct quadrant.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.tan()`](jax.numpy.tan.html#jax.numpy.tan "jax.numpy.tan"): compute the tangent of an angle

- [`jax.numpy.atan2()`](jax.numpy.atan2.html#jax.numpy.atan2 "jax.numpy.atan2"): the array API version of this function.

Examples

Consider a sequence of angles in radians between 0 and \\2\pi\\:

    >>> theta = jnp.linspace(-jnp.pi, jnp.pi, 9)
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(theta)
    [-3.14 -2.36 -1.57 -0.79  0.    0.79  1.57  2.36  3.14]

These angles can equivalently be represented by `(x,`` ``y)` coordinates on a unit circle:

    >>> x, y = jnp.cos(theta), jnp.sin(theta)

To reconstruct the input angle, we might be tempted to use the identity \\\tan(\theta) = y / x\\, and compute \\\theta = \tan^{-1}(y/x)\\. Unfortunately, this does not recover the input angle:

    >>> with jnp.printoptions(precision=2, suppress=True):  
    ...   print(jnp.arctan(y / x))
    [-0.    0.79  1.57 -0.79  0.    0.79  1.57 -0.79  0.  ]

The problem is that \\y/x\\ contains some ambiguity: although \\(y, x) = (-1, -1)\\ and \\(y, x) = (1, 1)\\ represent different points in Cartesian space, in both cases \\y / x = 1\\, and so the simple arctan approach loses information about which quadrant the angle lies in. [`arctan2()`](#jax.numpy.arctan2 "jax.numpy.arctan2") is built to address this:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...  print(jnp.arctan2(y, x))
    [ 3.14 -2.36 -1.57 -0.79  0.    0.79  1.57  2.36 -3.14]

The results match the input `theta`, except at the endpoints where \\+\pi\\ and \\-\pi\\ represent indistinguishable points on the unit circle. By convention, [`arctan2()`](#jax.numpy.arctan2 "jax.numpy.arctan2") always returns values between \\-\pi\\ and \\+\pi\\ inclusive.

[](jax.numpy.arctan.html "previous page")

previous

jax.numpy.arctan

[](jax.numpy.arctanh.html "next page")

next

jax.numpy.arctanh

Contents

- [`arctan2()`](#jax.numpy.arctan2)

By The JAX authors

© Copyright 2024, The JAX Authors.\
