- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.clamp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.clamp.rst "Download source file")
-  .pdf

# jax.lax.clamp

## Contents

- [`clamp()`](#jax.lax.clamp)

# jax.lax.clamp[\#](#jax-lax-clamp "Link to this heading")

jax.lax.clamp(*min*, *x*, *max*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1794-L1805)[\#](#jax.lax.clamp "Link to this definition")  
Elementwise clamp.

Returns \\\mathrm{clamp}(x) = \begin{cases} \mathit{min} & \text{if } x \< \mathit{min},\\ \mathit{max} & \text{if } x \> \mathit{max},\\ x & \text{otherwise} \end{cases}\\.

Parameters:  
- **min** (*ArrayLike*)

- **x** (*ArrayLike*)

- **max** (*ArrayLike*)

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.ceil.html "previous page")

previous

jax.lax.ceil

[](jax.lax.clz.html "next page")

next

jax.lax.clz

Contents

- [`clamp()`](#jax.lax.clamp)

By The JAX authors

© Copyright 2024, The JAX Authors.\
