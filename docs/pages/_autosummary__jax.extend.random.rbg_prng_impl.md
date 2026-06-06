- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.random` module](../jax.extend.random.html)
- jax.extend.random.rbg_prng_impl

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.random.rbg_prng_impl.rst "Download source file")
-  .pdf

# jax.extend.random.rbg_prng_impl

## Contents

- [`rbg_prng_impl`](#jax.extend.random.rbg_prng_impl)

# jax.extend.random.rbg_prng_impl[\#](#jax-extend-random-rbg-prng-impl "Link to this heading")

jax.extend.random.rbg_prng_impl *= ((4,), \<function \_rbg_seed\>, \<function \_rbg_split\>, \<function \_rbg_random_bits\>, \<function \_rbg_fold_in\>, 'rbg', 'rbg')*[\#](#jax.extend.random.rbg_prng_impl "Link to this definition")  
Specifies PRNG key shape and operations.

A PRNG implementation is determined by a key type `K` and a collection of functions that operate on such keys. The key type `K` is an array type with element type uint32 and shape specified by `key_shape`. The type signature of each operations is:

    seed :: int[] -> K
    fold_in :: K -> int[] -> K
    split[shape] :: K -> K[*shape]
    random_bits[shape, bit_width] :: K -> uint<bit_width>[*shape]

A PRNG implementation is adapted to an array-like object of keys `K` by the `PRNGKeyArray` class, which should be created via the `random_seed` function.

[](jax.extend.random.random_seed.html "previous page")

previous

jax.extend.random.random_seed

[](jax.extend.random.unsafe_rbg_prng_impl.html "next page")

next

jax.extend.random.unsafe_rbg_prng_impl

Contents

- [`rbg_prng_impl`](#jax.extend.random.rbg_prng_impl)

By The JAX authors

© Copyright 2024, The JAX Authors.\
