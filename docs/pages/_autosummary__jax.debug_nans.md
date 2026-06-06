- [](../index.html)
- [API Reference](../jax.html)
- jax.debug_nans

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.debug_nans.rst "Download source file")
-  .pdf

# jax.debug_nans

## Contents

- [`debug_nans`](#jax.debug_nans)

# jax.debug_nans[\#](#jax-debug-nans "Link to this heading")

jax.debug_nans *= \<jax.\_src.config.State object\>*[\#](#jax.debug_nans "Link to this definition")  
Context manager for jax_debug_nans config option.

Add nan checks to every operation. When a nan is detected on the output of a jit-compiled computation, call into the un-compiled version in an attempt to more precisely identify the operation which produced the nan.

Parameters:  
**new_val** (*Any*)

[](jax.checking_leaks.html "previous page")

previous

jax.checking_leaks

[](jax.debug_infs.html "next page")

next

jax.debug_infs

Contents

- [`debug_nans`](#jax.debug_nans)

By The JAX authors

© Copyright 2024, The JAX Authors.\
