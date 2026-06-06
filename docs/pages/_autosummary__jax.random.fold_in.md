- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.fold_in

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.fold_in.rst "Download source file")
-  .pdf

# jax.random.fold_in

## Contents

- [`fold_in()`](#jax.random.fold_in)

# jax.random.fold_in[\#](#jax-random-fold-in "Link to this heading")

jax.random.fold_in(*key*, *data*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L289-L306)[\#](#jax.random.fold_in "Link to this definition")  
Folds in data to a PRNG key to form a new PRNG key.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key (from `key`, `split`, `fold_in`).

- **data** (*IntegerArray*) – a 32-bit integer representing data to be folded into the key.

Returns:  
A new PRNG key that is a deterministic function of the inputs and is statistically safe for producing a stream of new pseudo-random values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.wrap_key_data.html "previous page")

previous

jax.random.wrap_key_data

[](jax.random.split.html "next page")

next

jax.random.split

Contents

- [`fold_in()`](#jax.random.fold_in)

By The JAX authors

© Copyright 2024, The JAX Authors.\
