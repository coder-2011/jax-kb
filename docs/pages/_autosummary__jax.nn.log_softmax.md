- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.log_softmax

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.log_softmax.rst "Download source file")
-  .pdf

# jax.nn.log_softmax

## Contents

- [`log_softmax()`](#jax.nn.log_softmax)

# jax.nn.log_softmax[\#](#jax-nn-log-softmax "Link to this heading")

jax.nn.log_softmax(*x*, *axis=-1*, *where=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L535-L575)[\#](#jax.nn.log_softmax "Link to this definition")  
Log-Softmax function.

Computes the logarithm of the `softmax` function, which rescales elements to the range \\\[-\infty, 0)\\.

\\\mathrm{log\\softmax}(x)\_i = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)} \right)\\

Parameters:  
- **x** (*ArrayLike*) – input array

- **axis** (*Axis*) – the axis or axes along which the `log_softmax` should be computed. Either an integer, tuple of integers, or `None` (all axes).

- **where** (*ArrayLike* *\|* *None*) – Elements to include in the `log_softmax`. The output for any masked-out element is minus infinity.

Returns:  
An array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

If any input values are `+inf`, the result will be all `NaN`: this reflects the fact that `inf`` ``/`` ``inf` is not well-defined in the context of floating-point math.

See also

[`softmax()`](jax.nn.softmax.html#jax.nn.softmax "jax.nn.softmax")

[](jax.nn.softmax.html "previous page")

previous

jax.nn.softmax

[](jax.nn.logmeanexp.html "next page")

next

jax.nn.logmeanexp

Contents

- [`log_softmax()`](#jax.nn.log_softmax)

By The JAX authors

© Copyright 2024, The JAX Authors.\
