- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.PRNGKey

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.PRNGKey.rst "Download source file")
-  .pdf

# jax.random.PRNGKey

## Contents

- [`PRNGKey()`](#jax.random.PRNGKey)

# jax.random.PRNGKey[\#](#jax-random-prngkey "Link to this heading")

jax.random.PRNGKey(*seed*, *\**, *impl=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L260-L287)[\#](#jax.random.PRNGKey "Link to this definition")  
Create a legacy PRNG key given an integer seed.

This function produces old-style legacy PRNG keys, which are arrays of dtype `uint32`. For more, see the note in the [PRNG keys](https://docs.jax.dev/en/latest/jax.random.html#prng-keys) section. When possible, [`jax.random.key()`](jax.random.key.html#jax.random.key "jax.random.key") is recommended for use instead.

The resulting key does not carry a PRNG implementation. The returned key matches the implementation given by the optional `impl` argument or, otherwise, determined by the `jax_default_prng_impl` config flag. Callers must ensure that same implementation is set as the default when passing this key as an argument to other functions (such as `jax.random.split` and `jax.random.normal`).

Parameters:  
- **seed** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *ArrayLike*) – a 64- or 32-bit integer used as the value of the key.

- **impl** (*PRNGSpecDesc* *\|* *None*) – optional string specifying the PRNG implementation (e.g. `'threefry2x32'`)

Returns:  
A PRNG key, consumable by random functions as well as `split` and `fold_in`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.clone.html "previous page")

previous

jax.random.clone

[](jax.random.ball.html "next page")

next

jax.random.ball

Contents

- [`PRNGKey()`](#jax.random.PRNGKey)

By The JAX authors

© Copyright 2024, The JAX Authors.\
