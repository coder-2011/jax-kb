- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.log_softmax

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.log_softmax.rst "Download source file")
-  .pdf

# jax.scipy.special.log_softmax

## Contents

- [`log_softmax()`](#jax.scipy.special.log_softmax)

# jax.scipy.special.log_softmax[\#](#jax-scipy-special-log-softmax "Link to this heading")

jax.scipy.special.log_softmax(*x*, */*, *\**, *axis=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L3894-L3927)[\#](#jax.scipy.special.log_softmax "Link to this definition")  
Log-Softmax function.

JAX implementation of [`scipy.special.log_softmax()`](https://scipy.github.io/devdocs/reference/generated/scipy.special.log_softmax.html#scipy.special.log_softmax "(in SciPy v1.19.0.dev)")

Computes the logarithm of the `softmax` function, which rescales elements to the range \\\[-\infty, 0)\\.

\\\mathrm{log\\softmax}(x)\_i = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)} \right)\\

Parameters:  
- **x** (*ArrayLike*) – input array

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]* *\|* *None*) – the axis or axes along which the `log_softmax` should be computed. `None` means all axes.

Returns:  
An array of the same shape as `x`

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

If any input values are `+inf`, the result will be all `NaN`: this reflects the fact that `inf`` ``/`` ``inf` is not well-defined in the context of floating-point math.

See also

[`softmax()`](jax.scipy.special.softmax.html#jax.scipy.special.softmax "jax.scipy.special.softmax")

[](jax.scipy.special.log_ndtr.html "previous page")

previous

jax.scipy.special.log_ndtr

[](jax.scipy.special.logit.html "next page")

next

jax.scipy.special.logit

Contents

- [`log_softmax()`](#jax.scipy.special.log_softmax)

By The JAX authors

© Copyright 2024, The JAX Authors.\
