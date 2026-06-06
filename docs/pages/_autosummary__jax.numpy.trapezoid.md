- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.trapezoid

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.trapezoid.rst "Download source file")
-  .pdf

# jax.numpy.trapezoid

## Contents

- [`trapezoid()`](#jax.numpy.trapezoid)

# jax.numpy.trapezoid[\#](#jax-numpy-trapezoid "Link to this heading")

jax.numpy.trapezoid(*y*, *x=None*, *dx=1.0*, *axis=-1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L6443-L6508)[\#](#jax.numpy.trapezoid "Link to this definition")  
Integrate along the given axis using the composite trapezoidal rule.

JAX implementation of [`numpy.trapezoid()`](https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html#numpy.trapezoid "(in NumPy v2.4)")

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

Examples

Integrate over a regular grid, with spacing 1.0:

    >>> y = jnp.array([1, 2, 3, 2, 3, 2, 1])
    >>> jnp.trapezoid(y, dx=1.0)
    Array(13., dtype=float32)

Integrate over an irregular grid:

    >>> x = jnp.array([0, 2, 5, 7, 10, 15, 20])
    >>> jnp.trapezoid(y, x)
    Array(43., dtype=float32)

Approximate \\\int_0^{2\pi} \sin^2(x)dx\\, which equals \\\pi\\:

    >>> x = jnp.linspace(0, 2 * jnp.pi, 1000)
    >>> y = jnp.sin(x) ** 2
    >>> result = jnp.trapezoid(y, x)
    >>> jnp.allclose(result, jnp.pi)
    Array(True, dtype=bool)

[](jax.numpy.trace.html "previous page")

previous

jax.numpy.trace

[](jax.numpy.transpose.html "next page")

next

jax.numpy.transpose

Contents

- [`trapezoid()`](#jax.numpy.trapezoid)

By The JAX authors

© Copyright 2024, The JAX Authors.\
