- [](index.html)
- [API Reference](jax.html)
- [`jax.experimental` module](jax.experimental.html)
- [`jax.experimental.pallas` module](jax.experimental.pallas.html)
- `jax.experimental.pallas.triton` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.experimental.pallas.triton.rst "Download source file")
-  .pdf

# jax.experimental.pallas.triton module

## Contents

- [Classes](#classes)
- [Functions](#functions)

# `jax.experimental.pallas.triton` module[\#](#module-jax.experimental.pallas.triton "Link to this heading")

Triton-specific Pallas APIs.

## Classes[\#](#classes "Link to this heading")

|  |  |
|----|----|
| [`CompilerParams`](_autosummary/jax.experimental.pallas.triton.CompilerParams.html#jax.experimental.pallas.triton.CompilerParams "jax.experimental.pallas.triton.CompilerParams")(\[num_warps, num_stages\]) | Compiler parameters for Triton. |

## Functions[\#](#functions "Link to this heading")

|  |  |
|----|----|
| [`atomic_and`](_autosummary/jax.experimental.pallas.triton.atomic_and.html#jax.experimental.pallas.triton.atomic_and "jax.experimental.pallas.triton.atomic_and")(x_ref_or_view, idx, val, \*\[, mask\]) | Atomically computes `x_ref_or_view[idx]`` ``&=`` ``val`. |
| [`atomic_add`](_autosummary/jax.experimental.pallas.triton.atomic_add.html#jax.experimental.pallas.triton.atomic_add "jax.experimental.pallas.triton.atomic_add")(x_ref_or_view, idx, val, \*\[, mask\]) | Atomically computes `x_ref_or_view[idx]`` ``+=`` ``val`. |
| [`atomic_cas`](_autosummary/jax.experimental.pallas.triton.atomic_cas.html#jax.experimental.pallas.triton.atomic_cas "jax.experimental.pallas.triton.atomic_cas")(ref, cmp, val) | Performs an atomic compare-and-swap of the value in the ref with the |
| [`atomic_max`](_autosummary/jax.experimental.pallas.triton.atomic_max.html#jax.experimental.pallas.triton.atomic_max "jax.experimental.pallas.triton.atomic_max")(x_ref_or_view, idx, val, \*\[, mask\]) | Atomically computes `x_ref_or_view[idx]`` ``=`` ``max(x_ref_or_view[idx],`` ``val)`. |
| [`atomic_min`](_autosummary/jax.experimental.pallas.triton.atomic_min.html#jax.experimental.pallas.triton.atomic_min "jax.experimental.pallas.triton.atomic_min")(x_ref_or_view, idx, val, \*\[, mask\]) | Atomically computes `x_ref_or_view[idx]`` ``=`` ``min(x_ref_or_view[idx],`` ``val)`. |
| [`atomic_or`](_autosummary/jax.experimental.pallas.triton.atomic_or.html#jax.experimental.pallas.triton.atomic_or "jax.experimental.pallas.triton.atomic_or")(x_ref_or_view, idx, val, \*\[, mask\]) | Atomically computes `x_ref_or_view[idx]`` ``|=`` ``val`. |
| [`atomic_xchg`](_autosummary/jax.experimental.pallas.triton.atomic_xchg.html#jax.experimental.pallas.triton.atomic_xchg "jax.experimental.pallas.triton.atomic_xchg")(x_ref_or_view, idx, val, \*\[, mask\]) | Atomically exchanges the given value with the value at the given index. |
| [`atomic_xor`](_autosummary/jax.experimental.pallas.triton.atomic_xor.html#jax.experimental.pallas.triton.atomic_xor "jax.experimental.pallas.triton.atomic_xor")(x_ref_or_view, idx, val, \*\[, mask\]) | Atomically computes `x_ref_or_view[idx]`` ``^=`` ``val`. |
| [`approx_tanh`](_autosummary/jax.experimental.pallas.triton.approx_tanh.html#jax.experimental.pallas.triton.approx_tanh "jax.experimental.pallas.triton.approx_tanh")(x) | Elementwise approximate hyperbolic tangent: \\\mathrm{tanh}(x)\\. |
| [`debug_barrier`](_autosummary/jax.experimental.pallas.triton.debug_barrier.html#jax.experimental.pallas.triton.debug_barrier "jax.experimental.pallas.triton.debug_barrier")() | Synchronizes all kernel executions in the grid. |
| [`elementwise_inline_asm`](_autosummary/jax.experimental.pallas.triton.elementwise_inline_asm.html#jax.experimental.pallas.triton.elementwise_inline_asm "jax.experimental.pallas.triton.elementwise_inline_asm")(asm, \*, args, ...) | Inline assembly applying an elementwise operation. |
| [`load`](_autosummary/jax.experimental.pallas.triton.load.html#jax.experimental.pallas.triton.load "jax.experimental.pallas.triton.load")(ref, \*\[, mask, other, cache_modifier, ...\]) | Loads an array from the given ref. |
| [`max_contiguous`](_autosummary/jax.experimental.pallas.triton.max_contiguous.html#jax.experimental.pallas.triton.max_contiguous "jax.experimental.pallas.triton.max_contiguous")(x, values) | A compiler hint that asserts the `values` first values of `x` are contiguous. |
| [`store`](_autosummary/jax.experimental.pallas.triton.store.html#jax.experimental.pallas.triton.store "jax.experimental.pallas.triton.store")(ref, val, \*\[, mask, eviction_policy\]) | Stores a value to the given ref. |

[](_autosummary/jax.experimental.pallas.mosaic_gpu.SMEM.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.SMEM

[](_autosummary/jax.experimental.pallas.triton.CompilerParams.html "next page")

next

jax.experimental.pallas.triton.CompilerParams

Contents

- [Classes](#classes)
- [Functions](#functions)

By The JAX authors

© Copyright 2024, The JAX Authors.\
