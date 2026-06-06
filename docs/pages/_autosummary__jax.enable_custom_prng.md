- [](../index.html)
- [API Reference](../jax.html)
- jax.enable_custom_prng

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.enable_custom_prng.rst "Download source file")
-  .pdf

# jax.enable_custom_prng

## Contents

- [`enable_custom_prng`](#jax.enable_custom_prng)

# jax.enable_custom_prng[\#](#jax-enable-custom-prng "Link to this heading")

jax.enable_custom_prng *= \<jax.\_src.config.State object\>*[\#](#jax.enable_custom_prng "Link to this definition")  
Context manager for jax_enable_custom_prng config option (transient).

Enables an internal upgrade that allows one to define custom pseudo-random number generator implementations. This will be enabled by default in future versions of JAX, at which point all uses of the flag will be considered deprecated (following the [API compatibility policy](https://docs.jax.dev/en/latest/api_compatibility.html)).

Parameters:  
**new_val** (*Any*)

[](jax.enable_checks.html "previous page")

previous

jax.enable_checks

[](jax.enable_x64.html "next page")

next

jax.enable_x64

Contents

- [`enable_custom_prng`](#jax.enable_custom_prng)

By The JAX authors

© Copyright 2024, The JAX Authors.\
