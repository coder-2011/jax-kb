- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.triton` module](../jax.experimental.pallas.triton.html)
- jax.experimental.pallas.triton.approx_tanh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.triton.approx_tanh.rst "Download source file")
-  .pdf

# jax.experimental.pallas.triton.approx_tanh

## Contents

- [`approx_tanh()`](#jax.experimental.pallas.triton.approx_tanh)

# jax.experimental.pallas.triton.approx_tanh[\#](#jax-experimental-pallas-triton-approx-tanh "Link to this heading")

jax.experimental.pallas.triton.approx_tanh(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/triton/primitives.py#L49-L80)[\#](#jax.experimental.pallas.triton.approx_tanh "Link to this definition")  
Elementwise approximate hyperbolic tangent: \\\mathrm{tanh}(x)\\.

See [https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-tanh](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-tanh).

Parameters:  
**x** ([*jax.Array*](jax.Array.html#jax.Array "jax.Array"))

Return type:  
[jax.Array](jax.Array.html#jax.Array "jax.Array")

[](jax.experimental.pallas.triton.atomic_xor.html "previous page")

previous

jax.experimental.pallas.triton.atomic_xor

[](jax.experimental.pallas.triton.debug_barrier.html "next page")

next

jax.experimental.pallas.triton.debug_barrier

Contents

- [`approx_tanh()`](#jax.experimental.pallas.triton.approx_tanh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
