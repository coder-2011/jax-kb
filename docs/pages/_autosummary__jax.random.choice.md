- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.choice

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.choice.rst "Download source file")
-  .pdf

# jax.random.choice

## Contents

- [`choice()`](#jax.random.choice)

# jax.random.choice[\#](#jax-random-choice "Link to this heading")

jax.random.choice(*key*, *a*, *shape=()*, *replace=True*, *p=None*, *axis=0*, *mode=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L824-L910)[\#](#jax.random.choice "Link to this definition")  
Generates a random sample from a given array.

Warning

If `p` has fewer non-zero elements than the requested number of samples, as specified in `shape`, and `replace=False`, the output of this function is ill-defined. Please make sure to use appropriate inputs.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **a** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *ArrayLike*) – array or int. If an ndarray, a random sample is generated from its elements. If an int, the random sample is generated as if a were arange(a).

- **shape** (*Shape*) – tuple of ints, optional. Output shape. If the given shape is, e.g., `(m,`` ``n)`, then `m`` ``*`` ``n` samples are drawn. Default is (), in which case a single value is returned.

- **replace** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – boolean. Whether the sample is with or without replacement. Default is True.

- **p** (*RealArray* *\|* *None*) – 1-D array-like, The probabilities associated with each entry in a. If not given the sample assumes a uniform distribution over all entries in a.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int, optional. The axis along which the selection is performed. The default, 0, selects by row.

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – optional, “high” or “low” for how many bits to use in the gumbel sampler when p is None and replace = False. The default is determined by the `use_high_dynamic_range_gumbel` config, which defaults to “low”. With mode=”low”, in float32 sampling will be biased for choices with probability less than about 1E-7; with mode=”high” this limit is pushed down to about 1E-14. mode=”high” approximately doubles the cost of sampling.

Returns:  
An array of shape shape containing samples from a.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.chisquare.html "previous page")

previous

jax.random.chisquare

[](jax.random.dirichlet.html "next page")

next

jax.random.dirichlet

Contents

- [`choice()`](#jax.random.choice)

By The JAX authors

© Copyright 2024, The JAX Authors.\
