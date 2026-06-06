- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.triton` module](../jax.experimental.pallas.triton.html)
- jax.experimental.pallas.triton.atomic_max

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.triton.atomic_max.rst "Download source file")
-  .pdf

# jax.experimental.pallas.triton.atomic_max

## Contents

- [`atomic_max()`](#jax.experimental.pallas.triton.atomic_max)

# jax.experimental.pallas.triton.atomic_max[\#](#jax-experimental-pallas-triton-atomic-max "Link to this heading")

jax.experimental.pallas.triton.atomic_max(*x_ref_or_view*, *idx*, *val*, *\**, *mask=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/triton/primitives.py#L536-L550)[\#](#jax.experimental.pallas.triton.atomic_max "Link to this definition")  
Atomically computes `x_ref_or_view[idx]`` ``=`` ``max(x_ref_or_view[idx],`` ``val)`.

Parameters:  
- **x_ref_or_view** – The ref to operate on.

- **idx** – The indexer to use.

- **mask** (*Any* *\|* *None*) – TO BE DOCUMENTED.

Returns:  
The value at the given index prior to the atomic operation.

[](jax.experimental.pallas.triton.atomic_cas.html "previous page")

previous

jax.experimental.pallas.triton.atomic_cas

[](jax.experimental.pallas.triton.atomic_min.html "next page")

next

jax.experimental.pallas.triton.atomic_min

Contents

- [`atomic_max()`](#jax.experimental.pallas.triton.atomic_max)

By The JAX authors

© Copyright 2024, The JAX Authors.\
