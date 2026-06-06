- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.multinomial

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.multinomial.rst "Download source file")
-  .pdf

# jax.random.multinomial

## Contents

- [`multinomial()`](#jax.random.multinomial)

# jax.random.multinomial[\#](#jax-random-multinomial "Link to this heading")

jax.random.multinomial(*key*, *n*, *p*, *\**, *shape=None*, *dtype=None*, *unroll=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L3337-L3395)[\#](#jax.random.multinomial "Link to this definition")  
Sample from a multinomial distribution.

The probability mass function is

\\f(x;n,p) = \frac{n!}{x_1! \ldots x_k!} p_1^{x_1} \ldots p_k^{x_k}\\

Parameters:  
- **key** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – PRNG key.

- **n** (*RealArray*) – number of trials. Should have shape broadcastable to `p.shape[:-1]`.

- **p** (*RealArray*) – probability of each outcome, with outcomes along the last axis.

- **shape** (*Shape* *\|* *None*) – optional, a tuple of nonnegative integers specifying the result batch shape, that is, the prefix of the result shape excluding the last axis. Must be broadcast-compatible with `p.shape[:-1]`. The default (None) produces a result shape equal to `p.shape`.

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **unroll** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – optional, unroll parameter passed to [`jax.lax.scan()`](jax.lax.scan.html#jax.lax.scan "jax.lax.scan") inside the implementation of this function.

Returns:  
An array of counts for each outcome with the specified dtype and with shape  
`p.shape` if `shape` is None, otherwise `shape`` ``+`` ``(p.shape[-1],)`.

[](jax.random.maxwell.html "previous page")

previous

jax.random.maxwell

[](jax.random.multivariate_normal.html "next page")

next

jax.random.multivariate_normal

Contents

- [`multinomial()`](#jax.random.multinomial)

By The JAX authors

© Copyright 2024, The JAX Authors.\
