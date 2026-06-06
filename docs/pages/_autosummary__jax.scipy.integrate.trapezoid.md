- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.integrate.trapezoid

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.integrate.trapezoid.rst "Download source file")
-  .pdf

# jax.scipy.integrate.trapezoid

## Contents

- [`trapezoid()`](#jax.scipy.integrate.trapezoid)

# jax.scipy.integrate.trapezoid[\#](#jax-scipy-integrate-trapezoid "Link to this heading")

jax.scipy.integrate.trapezoid(*y*, *x=None*, *dx=1.0*, *axis=-1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/integrate.py#L22-L68)[\#](#jax.scipy.integrate.trapezoid "Link to this definition")  
Integrate along the given axis using the composite trapezoidal rule.

JAX implementation of [`scipy.integrate.trapezoid()`](https://scipy.github.io/devdocs/reference/generated/scipy.integrate.trapezoid.html#scipy.integrate.trapezoid "(in SciPy v1.19.0.dev)")

The trapezoidal rule approximates the integral under a curve by summing the areas of trapezoids formed between adjacent data points.

Parameters:  
- **y** (*ArrayLike*) – array of data to integrate.

- **x** (*ArrayLike* *\|* *None*) – optional array of sample points corresponding to the `y` values. If not provided, `x` defaults to equally spaced with spacing given by `dx`.

- **dx** (*ArrayLike*) – The spacing between sample points when x is None (default: 1.0).

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The axis along which to integrate (default: -1)

Returns:  
The definite integral approximated by the trapezoidal rule.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.numpy.trapezoid()`](jax.numpy.trapezoid.html#jax.numpy.trapezoid "jax.numpy.trapezoid"): NumPy-style API for trapezoidal integration

Examples

Integrate over a regular grid, with spacing 1.0:

    >>> y = jnp.array([1, 2, 3, 2, 3, 2, 1])
    >>> jax.scipy.integrate.trapezoid(y, dx=1.0)
    Array(13., dtype=float32)

Integrate over an irregular grid:

    >>> x = jnp.array([0, 2, 5, 7, 10, 15, 20])
    >>> jax.scipy.integrate.trapezoid(y, x)
    Array(43., dtype=float32)

Approximate \\\int_0^{2\pi} \sin^2(x)dx\\, which equals \\\pi\\:

    >>> x = jnp.linspace(0, 2 * jnp.pi, 1000)
    >>> y = jnp.sin(x) ** 2
    >>> result = jax.scipy.integrate.trapezoid(y, x)
    >>> jnp.allclose(result, jnp.pi)
    Array(True, dtype=bool)

[](jax.scipy.fft.idctn.html "previous page")

previous

jax.scipy.fft.idctn

[](jax.scipy.interpolate.RegularGridInterpolator.html "next page")

next

jax.scipy.interpolate.RegularGridInterpolator

Contents

- [`trapezoid()`](#jax.scipy.integrate.trapezoid)

By The JAX authors

© Copyright 2024, The JAX Authors.\
