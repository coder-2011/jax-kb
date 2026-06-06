- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.standardize

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.standardize.rst "Download source file")
-  .pdf

# jax.nn.standardize

## Contents

- [`standardize()`](#jax.nn.standardize)

# jax.nn.standardize[\#](#jax-nn-standardize "Link to this heading")

jax.nn.standardize(*x*, *axis=-1*, *mean=None*, *variance=None*, *epsilon=1e-05*, *where=None*, *\**, *algorithm='fast'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L648-L712)[\#](#jax.nn.standardize "Link to this definition")  
Standardizes input to zero mean and unit variance.

The standardization is given by:

\\x\_{std} = \frac{x - \langle x\rangle}{\sqrt{\langle(x - \langle x\rangle)^2\rangle + \epsilon}}\\

where \\\langle x\rangle\\ indicates the mean of \\x\\, and \\\epsilon\\ is a small correction factor introduced to avoid division by zero.

Parameters:  
- **x** (*ArrayLike*) – input array to be standardized.

- **axis** (*Axis*) – integer, tuple of integers, or `None` (all axes), representing the axes along which to standardize. Defaults to the last axis (`-1`).

- **mean** (*ArrayLike* *\|* *None*) – optionally specify the mean used for standardization. If not specified, then `x.mean(axis,`` ``where=where)` will be used.

- **variance** (*ArrayLike* *\|* *None*) – optionally specify the variance used for standardization. If not specified, then `x.var(axis,`` ``where=where)` will be used.

- **epsilon** (*ArrayLike*) – correction factor added to variance to avoid division by zero; defaults to `1E-5`.

- **where** (*ArrayLike* *\|* *None*) – optional boolean mask specifying which elements to use when computing the mean and variance.

- **algorithm** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – variance computation algorithm. `"fast"` uses `mean(x^2)`` ``-`` ``mean(x)^2` which may be faster but can suffer from catastrophic cancellation and produce different results in eager vs JIT contexts. `"stable"` uses the two-pass formula `mean((x`` ``-`` ``mean(x))^2)` which is numerically stable. Default is `"fast"` for backward compatibility.

Returns:  
An array of the same shape as `x` containing the standardized input.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.nn.logsumexp.html "previous page")

previous

jax.nn.logsumexp

[](jax.nn.one_hot.html "next page")

next

jax.nn.one_hot

Contents

- [`standardize()`](#jax.nn.standardize)

By The JAX authors

© Copyright 2024, The JAX Authors.\
