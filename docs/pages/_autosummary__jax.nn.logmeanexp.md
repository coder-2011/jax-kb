- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.logmeanexp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.logmeanexp.rst "Download source file")
-  .pdf

# jax.nn.logmeanexp

## Contents

- [`logmeanexp()`](#jax.nn.logmeanexp)

# jax.nn.logmeanexp[\#](#jax-nn-logmeanexp "Link to this heading")

jax.nn.logmeanexp(*x*, *axis=None*, *where=None*, *keepdims=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L506-L533)[\#](#jax.nn.logmeanexp "Link to this definition")  
Log mean exp.

Computes the function:

\\\text{logmeanexp}(x) = \log \frac{1}{n} \sum\_{i=1}^n \exp x_i = \text{logsumexp}(x) - \log n\\

Parameters:  
- **x** (*ArrayLike*) – Input array.

- **axis** (*Axis*) – Axis or axes along which to reduce.

- **where** (*ArrayLike* *\|* *None*) – Elements to include in the reduction. Optional.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Preserve the dimensions of the input.

Returns:  
An array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.nn.logsumexp()`](jax.nn.logsumexp.html#jax.nn.logsumexp "jax.nn.logsumexp")

[](jax.nn.log_softmax.html "previous page")

previous

jax.nn.log_softmax

[](jax.nn.logsumexp.html "next page")

next

jax.nn.logsumexp

Contents

- [`logmeanexp()`](#jax.nn.logmeanexp)

By The JAX authors

© Copyright 2024, The JAX Authors.\
