- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.stage

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.stage.rst "Download source file")
-  .pdf

# jax.lax.stage

## Contents

- [`stage()`](#jax.lax.stage)

# jax.lax.stage[\#](#jax-lax-stage "Link to this heading")

jax.lax.stage(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1618-L1627)[\#](#jax.lax.stage "Link to this definition")  
Lifts a value into a trace.

This operation is logically the identity function that lifts a value, such as a Python scalar or numpy ndarray, into the active trace. If we are outside any active trace contexts, stage returns a JAX array.

Parameters:  
**x** (*ArrayLike*)

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.stack.html "previous page")

previous

jax.lax.stack

[](jax.lax.sub.html "next page")

next

jax.lax.sub

Contents

- [`stage()`](#jax.lax.stage)

By The JAX authors

© Copyright 2024, The JAX Authors.\
