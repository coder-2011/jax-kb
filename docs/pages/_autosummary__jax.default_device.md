- [](../index.html)
- [API Reference](../jax.html)
- jax.default_device

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.default_device.rst "Download source file")
-  .pdf

# jax.default_device

## Contents

- [`default_device`](#jax.default_device)

# jax.default_device[\#](#jax-default-device "Link to this heading")

jax.default_device *= \<jax.\_src.config.State object\>*[\#](#jax.default_device "Link to this definition")  
Context manager for jax_default_device config option.

Configure the default device for JAX operations. Set to a Device object (e.g. `jax.devices("cpu")[0]`) to use that Device as the default device for JAX operations and jit’d function calls (there is no effect on multi-device computations, e.g. pmapped function calls). Set to None to use the system default device.

Parameters:  
**new_val** (*Any*)

[](jax.debug_infs.html "previous page")

previous

jax.debug_infs

[](jax.default_matmul_precision.html "next page")

next

jax.default_matmul_precision

Contents

- [`default_device`](#jax.default_device)

By The JAX authors

© Copyright 2024, The JAX Authors.\
