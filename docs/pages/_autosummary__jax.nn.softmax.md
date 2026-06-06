- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.softmax

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.softmax.rst "Download source file")
-  .pdf

# jax.nn.softmax

## Contents

- [`softmax()`](#jax.nn.softmax)

# jax.nn.softmax[\#](#jax-nn-softmax "Link to this heading")

jax.nn.softmax(*x*, *axis=-1*, *where=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L579-L612)[\#](#jax.nn.softmax "Link to this definition")  
Softmax function.

Computes the function which rescales elements to the range \\\[0, 1\]\\ such that the elements along `axis` sum to \\1\\.

\\\mathrm{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}\\

Parameters:  
- **x** (*ArrayLike*) – input array

- **axis** (*Axis*) – the axis or axes along which the softmax should be computed. The softmax output summed across these dimensions should sum to \\1\\. Either an integer, tuple of integers, or `None` (all axes).

- **where** (*ArrayLike* *\|* *None*) – Elements to include in the `softmax`. The output for any masked-out element is zero.

Returns:  
An array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

If any input values are `+inf`, the result will be all `NaN`: this reflects the fact that `inf`` ``/`` ``inf` is not well-defined in the context of floating-point math.

See also

[`log_softmax()`](jax.nn.log_softmax.html#jax.nn.log_softmax "jax.nn.log_softmax")

[](jax.nn.identity.html "previous page")

previous

jax.nn.identity

[](jax.nn.log_softmax.html "next page")

next

jax.nn.log_softmax

Contents

- [`softmax()`](#jax.nn.softmax)

By The JAX authors

© Copyright 2024, The JAX Authors.\
