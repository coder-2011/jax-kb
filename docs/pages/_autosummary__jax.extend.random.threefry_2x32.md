- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.random` module](../jax.extend.random.html)
- jax.extend.random.threefry_2x32

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.random.threefry_2x32.rst "Download source file")
-  .pdf

# jax.extend.random.threefry_2x32

## Contents

- [`threefry_2x32()`](#jax.extend.random.threefry_2x32)

# jax.extend.random.threefry_2x32[\#](#jax-extend-random-threefry-2x32 "Link to this heading")

jax.extend.random.threefry_2x32(*keypair*, *count*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/threefry2x32.py#L235-L280)[\#](#jax.extend.random.threefry_2x32 "Link to this definition")  
Apply the Threefry 2x32 hash.

Parameters:  
- **keypair** – a pair of 32bit unsigned integers used for the key.

- **count** – an array of dtype uint32 used for the counts.

Returns:  
An array of dtype uint32 with the same shape as count.

[](jax.extend.random.threefry2x32_p.html "previous page")

previous

jax.extend.random.threefry2x32_p

[](jax.extend.random.threefry_prng_impl.html "next page")

next

jax.extend.random.threefry_prng_impl

Contents

- [`threefry_2x32()`](#jax.extend.random.threefry_2x32)

By The JAX authors

© Copyright 2024, The JAX Authors.\
