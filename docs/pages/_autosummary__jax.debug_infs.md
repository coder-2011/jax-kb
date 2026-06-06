- [](../index.html)
- [API Reference](../jax.html)
- jax.debug_infs

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.debug_infs.rst "Download source file")
-  .pdf

# jax.debug_infs

## Contents

- [`debug_infs`](#jax.debug_infs)

# jax.debug_infs[\#](#jax-debug-infs "Link to this heading")

jax.debug_infs *= \<jax.\_src.config.State object\>*[\#](#jax.debug_infs "Link to this definition")  
Context manager for jax_debug_infs config option.

Add inf checks to every operation. When an inf is detected on the output of a jit-compiled computation, call into the un-compiled version in an attempt to more precisely identify the operation which produced the inf.

Parameters:  
**new_val** (*Any*)

[](jax.debug_nans.html "previous page")

previous

jax.debug_nans

[](jax.default_device.html "next page")

next

jax.default_device

Contents

- [`debug_infs`](#jax.debug_infs)

By The JAX authors

© Copyright 2024, The JAX Authors.\
