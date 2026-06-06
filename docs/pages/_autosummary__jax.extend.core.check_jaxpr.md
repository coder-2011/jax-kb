- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.core` module](../jax.extend.core.html)
- jax.extend.core.check_jaxpr

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.core.check_jaxpr.rst "Download source file")
-  .pdf

# jax.extend.core.check_jaxpr

## Contents

- [`check_jaxpr()`](#jax.extend.core.check_jaxpr)

# jax.extend.core.check_jaxpr[\#](#jax-extend-core-check-jaxpr "Link to this heading")

jax.extend.core.check_jaxpr(*jaxpr*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py#L3500-L3538)[\#](#jax.extend.core.check_jaxpr "Link to this definition")  
Checks well-formedness of a jaxpr.

Specifically, check that: - variables that are read are bound beforehand - variables are typed equally throughout a jaxpr - variable type annotations are compatible with their binding expression

Raises JaxprTypeError if jaxpr is determined invalid. Returns None otherwise.

Parameters:  
**jaxpr** ([*Jaxpr*](jax.extend.core.Jaxpr.html#jax.extend.core.Jaxpr "jax.extend.core.Jaxpr"))

[](jax.extend.core.call_impl.html "previous page")

previous

jax.extend.core.call_impl

[](jax.extend.core.concrete_or_error.html "next page")

next

jax.extend.core.concrete_or_error

Contents

- [`check_jaxpr()`](#jax.extend.core.check_jaxpr)

By The JAX authors

© Copyright 2024, The JAX Authors.\
