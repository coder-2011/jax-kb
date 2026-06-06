- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.weibull_min

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.weibull_min.rst "Download source file")
-  .pdf

# jax.random.weibull_min

## Contents

- [`weibull_min()`](#jax.random.weibull_min)

# jax.random.weibull_min[\#](#jax-random-weibull-min "Link to this heading")

jax.random.weibull_min(*key*, *scale*, *concentration*, *shape=()*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L2596-L2630)[\#](#jax.random.weibull_min "Link to this definition")  
Sample from a Weibull distribution.

The values are distributed according to the probability density function:

\\f(x;\sigma,c) \propto x^{c - 1} \exp(-(x / \sigma)^c)\\

on the domain \\0 \< x \< \infty\\, where \\c \> 0\\ is the concentration parameter, and \\\sigma \> 0\\ is the scale parameter.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key.

- **scale** (*RealArray*) – The scale parameter of the distribution.

- **concentration** (*RealArray*) – The concentration parameter of the distribution.

- **shape** (*Shape*) – The shape added to the parameters loc and scale broadcastable shape.

- **dtype** (*DTypeLikeFloat* *\|* *None*) – The type used for samples.

Returns:  
A jnp.array of samples.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.wald.html "previous page")

previous

jax.random.wald

[](../jax.sharding.html "next page")

next

`jax.sharding` module

Contents

- [`weibull_min()`](#jax.random.weibull_min)

By The JAX authors

© Copyright 2024, The JAX Authors.\
