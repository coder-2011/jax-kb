- [](index.html)
- [API Reference](jax.html)
- `jax.ref` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.ref.rst "Download source file")
-  .pdf

# jax.ref module

## Contents

- [API](#api)

# `jax.ref` module[\#](#module-jax.ref "Link to this heading")

[`jax.ref`](#module-jax.ref "jax.ref") has the API for working with `ArrayRef`.

## API[\#](#api "Link to this heading")

|  |  |
|----|----|
| [`AbstractRef`](_autosummary/jax.ref.AbstractRef.html#jax.ref.AbstractRef "jax.ref.AbstractRef")(inner_aval\[, memory_space, kind\]) | Abstract mutable array reference. |
| [`Ref`](_autosummary/jax.ref.Ref.html#jax.ref.Ref "jax.ref.Ref")(aval, refs) | Mutable array reference. |
| [`freeze`](_autosummary/jax.ref.freeze.html#jax.ref.freeze "jax.ref.freeze")(ref) | Invalidate a given reference and return its final value. |
| [`get`](_autosummary/jax.ref.get.html#jax.ref.get "jax.ref.get")(ref\[, idx\]) | Read a value from an Ref. |
| [`new_ref`](_autosummary/jax.ref.new_ref.html#jax.ref.new_ref "jax.ref.new_ref")(init_val, \*\[, memory_space, kind\]) | Create a mutable array reference with initial value `init_val`. |
| [`set`](_autosummary/jax.ref.set.html#jax.ref.set "jax.ref.set")(ref, idx, value) | Set a value in an Ref in-place. |
| [`swap`](_autosummary/jax.ref.swap.html#jax.ref.swap "jax.ref.swap")(ref, idx, value\[, \_function_name\]) | Update an array value inplace while returning the previous value. |
| [`addupdate`](_autosummary/jax.ref.addupdate.html#jax.ref.addupdate "jax.ref.addupdate")(ref, idx, x) | Add to an element in an Ref in-place. |

[](_autosummary/jax.profiler.save_device_memory_profile.html "previous page")

previous

jax.profiler.save_device_memory_profile

[](_autosummary/jax.ref.AbstractRef.html "next page")

next

jax.ref.AbstractRef

Contents

- [API](#api)

By The JAX authors

© Copyright 2024, The JAX Authors.\
