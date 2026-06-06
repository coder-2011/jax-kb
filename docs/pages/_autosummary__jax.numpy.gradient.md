- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.gradient

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.gradient.rst "Download source file")
-  .pdf

# jax.numpy.gradient

## Contents

- [`gradient()`](#jax.numpy.gradient)

# jax.numpy.gradient[\#](#jax-numpy-gradient "Link to this heading")

jax.numpy.gradient(*f*, *\*varargs*, *axis=None*, *edge_order=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L1735-L1864)[\#](#jax.numpy.gradient "Link to this definition")  
Compute the numerical gradient of a sampled function.

JAX implementation of [`numpy.gradient()`](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html#numpy.gradient "(in NumPy v2.4)").

The gradient in `jnp.gradient` is computed using second-order finite differences across the array of sampled function values. This should not be confused with [`jax.grad()`](jax.grad.html#jax.grad "jax.grad"), which computes a precise gradient of a callable function via [automatic differentiation](../automatic-differentiation.html#automatic-differentiation).

Parameters:  
- **f** (*ArrayLike*) – *N*-dimensional array of function values.

- **varargs** (*ArrayLike*) –

  optional list of scalars or arrays specifying spacing of function evaluations. Options are:

  - not specified: unit spacing in all dimensions.

  - a single scalar: constant spacing in all dimensions.

  - *N* values: specify different spacing in each dimension:

    - scalar values indicate constant spacing in that dimension.

    - array values must match the length of the corresponding dimension, and specify the coordinates at which `f` is evaluated.

- **edge_order** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – not implemented in JAX

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – integer or tuple of integers specifying the axis along which to compute the gradient. If None (default) calculates the gradient along all axes.

Returns:  
an array or tuple of arrays containing the numerical gradient along each specified axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array") \| [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array")\]

See also

- [`jax.grad()`](jax.grad.html#jax.grad "jax.grad"): automatic differentiation of a function with a single output.

Examples

Comparing numerical and automatic differentiation of a simple function:

    >>> def f(x):
    ...   return jnp.sin(x) * jnp.exp(-x / 4)
    ...
    >>> def gradf_exact(x):
    ...   # exact analytical gradient of f(x)
    ...   return -f(x) / 4 + jnp.cos(x) * jnp.exp(-x / 4)
    ...
    >>> x = jnp.linspace(0, 5, 10)

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print("numerical gradient:", jnp.gradient(f(x), x))
    ...   print("automatic gradient:", jax.vmap(jax.grad(f))(x))
    ...   print("exact gradient:    ", gradf_exact(x))
    ...
    numerical gradient: [ 0.83  0.61  0.18 -0.2  -0.43 -0.49 -0.39 -0.21 -0.02  0.08]
    automatic gradient: [ 1.    0.62  0.17 -0.23 -0.46 -0.51 -0.41 -0.21 -0.01  0.15]
    exact gradient:     [ 1.    0.62  0.17 -0.23 -0.46 -0.51 -0.41 -0.21 -0.01  0.15]

Notice that, as expected, the numerical gradient has some approximation error compared to the automatic gradient computed via [`jax.grad()`](jax.grad.html#jax.grad "jax.grad").

[](jax.numpy.get_printoptions.html "previous page")

previous

jax.numpy.get_printoptions

[](jax.numpy.greater.html "next page")

next

jax.numpy.greater

Contents

- [`gradient()`](#jax.numpy.gradient)

By The JAX authors

© Copyright 2024, The JAX Authors.\
