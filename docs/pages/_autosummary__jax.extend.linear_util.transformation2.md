- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.linear_util` module](../jax.extend.linear_util.html)
- jax.extend.linear_util.transformation2

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.linear_util.transformation2.rst "Download source file")
-  .pdf

# jax.extend.linear_util.transformation2

## Contents

- [`transformation2`](#jax.extend.linear_util.transformation2)

# jax.extend.linear_util.transformation2[\#](#jax-extend-linear-util-transformation2 "Link to this heading")

jax.extend.linear_util.transformation2 *= functools.partial(\<class 'functools.partial'\>, \<function transformation2\>)*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/linear_util.py#L238-L248)[\#](#jax.extend.linear_util.transformation2 "Link to this definition")  
Adds one more transformation to a WrappedFun.

Parameters:  
- **gen** – the transformation generator function

- **fun** ([*WrappedFun*](jax.extend.linear_util.WrappedFun.html#jax.extend.linear_util.WrappedFun "jax.extend.linear_util.WrappedFun")) – a WrappedFun on which to apply the transformation

- **gen_static_args** – static args for the generator function

Return type:  
[WrappedFun](jax.extend.linear_util.WrappedFun.html#jax.extend.linear_util.WrappedFun "jax.extend.linear_util.WrappedFun")

[](jax.extend.linear_util.transformation.html "previous page")

previous

jax.extend.linear_util.transformation

[](jax.extend.linear_util.transformation_with_aux.html "next page")

next

jax.extend.linear_util.transformation_with_aux

Contents

- [`transformation2`](#jax.extend.linear_util.transformation2)

By The JAX authors

© Copyright 2024, The JAX Authors.\
