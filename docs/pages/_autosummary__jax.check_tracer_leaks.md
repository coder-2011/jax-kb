- [](../index.html)
- [API Reference](../jax.html)
- jax.check_tracer_leaks

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.check_tracer_leaks.rst "Download source file")
-  .pdf

# jax.check_tracer_leaks

## Contents

- [`check_tracer_leaks`](#jax.check_tracer_leaks)

# jax.check_tracer_leaks[\#](#jax-check-tracer-leaks "Link to this heading")

jax.check_tracer_leaks *= \<jax.\_src.config.State object\>*[\#](#jax.check_tracer_leaks "Link to this definition")  
Context manager for jax_check_tracer_leaks config option.

Turn on checking for leaked tracers as soon as a trace completes. Enabling leak checking may have performance impacts: some caching is disabled, and other overheads may be added. Additionally, be aware that some Python debuggers can cause false positives, so it is recommended to disable any debuggers while leak checking is enabled.

Parameters:  
**new_val** (*Any*)

[](jax.config.html "previous page")

previous

jax.config

[](jax.checking_leaks.html "next page")

next

jax.checking_leaks

Contents

- [`check_tracer_leaks`](#jax.check_tracer_leaks)

By The JAX authors

© Copyright 2024, The JAX Authors.\
