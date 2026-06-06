- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.key

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.key.rst "Download source file")
-  .pdf

# jax.random.key

## Contents

- [`key()`](#jax.random.key)

# jax.random.key[\#](#jax-random-key "Link to this heading")

jax.random.key(*seed*, *\**, *impl=None*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L232-L259)[\#](#jax.random.key "Link to this definition")  
Create a pseudo-random number generator (PRNG) key given an integer seed.

The result is a scalar array containing a key, whose dtype indicates the default PRNG implementation, as determined by the optional `dtype` or `impl` argument or, otherwise, by the `jax_default_prng_impl` config flag at the time when this function is called.

Parameters:  
- **seed** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *ArrayLike*) – a 64- or 32-bit integer used as the value of the key.

- **impl** (*PRNGSpecDesc* *\|* *None*) – optional string specifying the PRNG implementation (e.g. `'threefry2x32'`). Deprecated in favor of `dtype`.

- **dtype** (*KeyDTypeLike* *\|* *None*) – optional dtype or string name specifying the PRNG implementation (e.g. `jax.random.key_dtype('threefry2x32')` or `'threefry2x32'`).

Returns:  
A scalar PRNG key array, consumable by random functions as well as `split` and `fold_in`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](../jax.random.html "previous page")

previous

`jax.random` module

[](jax.random.key_dtype.html "next page")

next

jax.random.key_dtype

Contents

- [`key()`](#jax.random.key)

By The JAX authors

© Copyright 2024, The JAX Authors.\
