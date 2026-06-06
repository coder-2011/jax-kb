- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.interp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.interp.rst "Download source file")
-  .pdf

# jax.numpy.interp

## Contents

- [`interp()`](#jax.numpy.interp)

# jax.numpy.interp[\#](#jax-numpy-interp "Link to this heading")

jax.numpy.interp(*x*, *xp*, *fp*, *left=None*, *right=None*, *period=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L2652-L2714)[\#](#jax.numpy.interp "Link to this definition")  
One-dimensional linear interpolation.

JAX implementation of [`numpy.interp()`](https://numpy.org/doc/stable/reference/generated/numpy.interp.html#numpy.interp "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – N-dimensional array of x coordinates at which to evaluate the interpolation.

- **xp** (*ArrayLike*) – one-dimensional sorted array of points to be interpolated.

- **fp** (*ArrayLike*) – array of shape `xp.shape` containing the function values associated with `xp`.

- **left** (*ArrayLike* *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – specify how to handle points `x`` ``<`` ``xp[0]`. Default is to return `fp[0]`. If `left` is a scalar value, it will return this value. if `left` is the string `"extrapolate"`, then the value will be determined by linear extrapolation. `left` is ignored if `period` is specified.

- **right** (*ArrayLike* *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – specify how to handle points `x`` ``>`` ``xp[-1]`. Default is to return `fp[-1]`. If `right` is a scalar value, it will return this value. if `right` is the string `"extrapolate"`, then the value will be determined by linear extrapolation. `right` is ignored if `period` is specified.

- **period** (*ArrayLike* *\|* *None*) – optionally specify the period for the *x* coordinates, for e.g. interpolation in angular space.

Returns:  
an array of shape `x.shape` containing the interpolated function at values `x`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> xp = jnp.arange(10)
    >>> fp = 2 * xp
    >>> x = jnp.array([0.5, 2.0, 3.5])
    >>> interp(x, xp, fp)
    Array([1., 4., 7.], dtype=float32)

Unless otherwise specified, extrapolation will be constant:

    >>> x = jnp.array([-10., 10.])
    >>> interp(x, xp, fp)
    Array([ 0., 18.], dtype=float32)

Use `"extrapolate"` mode for linear extrapolation:

    >>> interp(x, xp, fp, left='extrapolate', right='extrapolate')
    Array([-20.,  20.], dtype=float32)

For periodic interpolation, specify the `period`:

    >>> xp = jnp.array([0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2])
    >>> fp = jnp.sin(xp)
    >>> x = 2 * jnp.pi  # note: not in input array
    >>> jnp.interp(x, xp, fp, period=2 * jnp.pi)
    Array(0., dtype=float32)

[](jax.numpy.integer.html "previous page")

previous

jax.numpy.integer

[](jax.numpy.intersect1d.html "next page")

next

jax.numpy.intersect1d

Contents

- [`interp()`](#jax.numpy.interp)

By The JAX authors

© Copyright 2024, The JAX Authors.\
