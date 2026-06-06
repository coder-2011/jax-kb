- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.softmax

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.softmax.rst "Download source file")
-  .pdf

# jax.scipy.special.softmax

## Contents

- [`softmax()`](#jax.scipy.special.softmax)

# jax.scipy.special.softmax[\#](#jax-scipy-special-softmax "Link to this heading")

jax.scipy.special.softmax(*x*, */*, *\**, *axis=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L3859-L3892)[\#](#jax.scipy.special.softmax "Link to this definition")  
Softmax function.

JAX implementation of [`scipy.special.softmax()`](https://scipy.github.io/devdocs/reference/generated/scipy.special.softmax.html#scipy.special.softmax "(in SciPy v1.19.0.dev)").

Computes the function which rescales elements to the range \\\[0, 1\]\\ such that the elements along `axis` sum to \\1\\.

\\\mathrm{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}\\

Parameters:  
- **x** (*ArrayLike*) – input array

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]* *\|* *None*) – the axis or axes along which the softmax should be computed. The softmax output summed across these dimensions should sum to \\1\\. `None` means all axes.

Returns:  
An array of the same shape as `x`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

If any input values are `+inf`, the result will be all `NaN`: this reflects the fact that `inf`` ``/`` ``inf` is not well-defined in the context of floating-point math.

See also

[`log_softmax()`](jax.scipy.special.log_softmax.html#jax.scipy.special.log_softmax "jax.scipy.special.log_softmax")

[](jax.scipy.special.sici.html "previous page")

previous

jax.scipy.special.sici

[](jax.scipy.special.spence.html "next page")

next

jax.scipy.special.spence

Contents

- [`softmax()`](#jax.scipy.special.softmax)

By The JAX authors

© Copyright 2024, The JAX Authors.\
