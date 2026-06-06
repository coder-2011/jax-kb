- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.core` module](../jax.extend.core.html)
- jax.extend.core.subjaxprs

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.core.subjaxprs.rst "Download source file")
-  .pdf

# jax.extend.core.subjaxprs

## Contents

- [`subjaxprs()`](#jax.extend.core.subjaxprs)

# jax.extend.core.subjaxprs[\#](#jax-extend-core-subjaxprs "Link to this heading")

jax.extend.core.subjaxprs(*jaxpr*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py#L238-L244)[\#](#jax.extend.core.subjaxprs "Link to this definition")  
Generator for all subjaxprs found in the params of jaxpr.eqns. Does not descend recursively into the found subjaxprs.

Parameters:  
**jaxpr** ([*Jaxpr*](jax.extend.core.Jaxpr.html#jax.extend.core.Jaxpr "jax.extend.core.Jaxpr"))

Return type:  
Iterator\[[Jaxpr](jax.extend.core.Jaxpr.html#jax.extend.core.Jaxpr "jax.extend.core.Jaxpr")\]

[](jax.extend.core.set_current_trace.html "previous page")

previous

jax.extend.core.set_current_trace

[](jax.extend.core.take_current_trace.html "next page")

next

jax.extend.core.take_current_trace

Contents

- [`subjaxprs()`](#jax.extend.core.subjaxprs)

By The JAX authors

© Copyright 2024, The JAX Authors.\
