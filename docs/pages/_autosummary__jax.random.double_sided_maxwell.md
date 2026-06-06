- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.double_sided_maxwell

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.double_sided_maxwell.rst "Download source file")
-  .pdf

# jax.random.double_sided_maxwell

## Contents

- [`double_sided_maxwell()`](#jax.random.double_sided_maxwell)

# jax.random.double_sided_maxwell[\#](#jax-random-double-sided-maxwell "Link to this heading")

jax.random.double_sided_maxwell(*key*, *loc*, *scale*, *shape=()*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L2544-L2578)[\#](#jax.random.double_sided_maxwell "Link to this definition")  
Sample from a double sided Maxwell distribution.

The values are distributed according to the probability density function:

\\f(x;\mu,\sigma) \propto z^2 e^{-z^2 / 2}\\

where \\z = (x - \mu) / \sigma\\, with the center \\\mu\\ specified by `loc` and the scale \\\sigma\\ specified by `scale`.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key.

- **loc** (*RealArray*) – The location parameter of the distribution.

- **scale** (*RealArray*) – The scale parameter of the distribution.

- **shape** (*Shape*) – The shape added to the parameters loc and scale broadcastable shape.

- **dtype** (*DTypeLikeFloat* *\|* *None*) – The type used for samples.

Returns:  
A jnp.array of samples.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.dirichlet.html "previous page")

previous

jax.random.dirichlet

[](jax.random.exponential.html "next page")

next

jax.random.exponential

Contents

- [`double_sided_maxwell()`](#jax.random.double_sided_maxwell)

By The JAX authors

© Copyright 2024, The JAX Authors.\
