- [](../index.html)
- [API Reference](../jax.html)
- jax.checking_leaks

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.checking_leaks.rst "Download source file")
-  .pdf

# jax.checking_leaks

## Contents

- [`checking_leaks`](#jax.checking_leaks)

# jax.checking_leaks[\#](#jax-checking-leaks "Link to this heading")

jax.checking_leaks *= functools.partial(\<jax.\_src.config.State object\>, True)*[\#](#jax.checking_leaks "Link to this definition")  
Context manager for jax_check_tracer_leaks config option.

Turn on checking for leaked tracers as soon as a trace completes. Enabling leak checking may have performance impacts: some caching is disabled, and other overheads may be added. Additionally, be aware that some Python debuggers can cause false positives, so it is recommended to disable any debuggers while leak checking is enabled.

[](jax.check_tracer_leaks.html "previous page")

previous

jax.check_tracer_leaks

[](jax.debug_nans.html "next page")

next

jax.debug_nans

Contents

- [`checking_leaks`](#jax.checking_leaks)

By The JAX authors

© Copyright 2024, The JAX Authors.\
