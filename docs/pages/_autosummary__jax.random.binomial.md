- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.binomial

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.binomial.rst "Download source file")
-  .pdf

# jax.random.binomial

## Contents

- [`binomial()`](#jax.random.binomial)

# jax.random.binomial[\#](#jax-random-binomial "Link to this heading")

jax.random.binomial(*key*, *n*, *p*, *shape=None*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L3282-L3327)[\#](#jax.random.binomial "Link to this definition")  
Sample Binomial random values with given shape and float dtype.

The values are returned according to the probability mass function:

\\f(k;n,p) = \binom{n}{k}p^k(1-p)^{n-k}\\

on the domain \\0 \< p \< 1\\, and where \\n\\ is a nonnegative integer representing the number of trials and \\p\\ is a float representing the probability of success of an individual trial.

Parameters:  
- **key** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – a PRNG key used as the random key.

- **n** (*RealArray*) – a float or array of floats broadcast-compatible with `shape` representing the number of trials.

- **p** (*RealArray*) – a float or array of floats broadcast-compatible with `shape` representing the probability of success of an individual trial.

- **shape** (*Shape* *\|* *None*) – optional, a tuple of nonnegative integers specifying the result shape. Must be broadcast-compatible with `n` and `p`. The default (None) produces a result shape equal to `np.broadcast(n,`` ``p).shape`.

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

Returns:  
A random array with the specified dtype and with shape given by `np.broadcast(n,`` ``p).shape`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.beta.html "previous page")

previous

jax.random.beta

[](jax.random.bits.html "next page")

next

jax.random.bits

Contents

- [`binomial()`](#jax.random.binomial)

By The JAX authors

© Copyright 2024, The JAX Authors.\
