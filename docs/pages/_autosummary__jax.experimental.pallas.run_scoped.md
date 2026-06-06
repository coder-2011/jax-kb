- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.run_scoped

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.run_scoped.rst "Download source file")
-  .pdf

# jax.experimental.pallas.run_scoped

## Contents

- [`run_scoped()`](#jax.experimental.pallas.run_scoped)

# jax.experimental.pallas.run_scoped[\#](#jax-experimental-pallas-run-scoped "Link to this heading")

jax.experimental.pallas.run_scoped(*f*, *\*types*, *collective_axes=()*, *\*\*kw_types*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/primitives.py#L670-L715)[\#](#jax.experimental.pallas.run_scoped "Link to this definition")  
Calls the function with allocated references and returns the result.

The positional and keyword arguments describe which reference types to allocate for each argument. Each backend has its own set of reference types in addition to `jax.experimental.pallas.MemoryRef`.

When `collective_axes` is specified, the same allocation will be returned for all programs that only differ in their program ids along the collective axes. It is an error not to call the same `run_scoped` in all programs along that axis.

Parameters:  
- **f** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]*)

- **types** (*Any*)

- **collective_axes** (*Hashable* *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Hashable,* *...\]*)

- **kw_types** (*Any*)

Return type:  
Any

[](jax.experimental.pallas.multiple_of.html "previous page")

previous

jax.experimental.pallas.multiple_of

[](jax.experimental.pallas.when.html "next page")

next

jax.experimental.pallas.when

Contents

- [`run_scoped()`](#jax.experimental.pallas.run_scoped)

By The JAX authors

© Copyright 2024, The JAX Authors.\
