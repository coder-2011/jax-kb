- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.triton` module](../jax.experimental.pallas.triton.html)
- jax.experimental.pallas.triton.atomic_or

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.triton.atomic_or.rst "Download source file")
-  .pdf

# jax.experimental.pallas.triton.atomic_or

## Contents

- [`atomic_or()`](#jax.experimental.pallas.triton.atomic_or)

# jax.experimental.pallas.triton.atomic_or[\#](#jax-experimental-pallas-triton-atomic-or "Link to this heading")

jax.experimental.pallas.triton.atomic_or(*x_ref_or_view*, *idx*, *val*, *\**, *mask=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/triton/primitives.py#L584-L598)[\#](#jax.experimental.pallas.triton.atomic_or "Link to this definition")  
Atomically computes `x_ref_or_view[idx]`` ``|=`` ``val`.

Parameters:  
- **x_ref_or_view** – The ref to operate on.

- **idx** – The indexer to use.

- **mask** (*Any* *\|* *None*) – TO BE DOCUMENTED.

Returns:  
The value at the given index prior to the atomic operation.

[](jax.experimental.pallas.triton.atomic_min.html "previous page")

previous

jax.experimental.pallas.triton.atomic_min

[](jax.experimental.pallas.triton.atomic_xchg.html "next page")

next

jax.experimental.pallas.triton.atomic_xchg

Contents

- [`atomic_or()`](#jax.experimental.pallas.triton.atomic_or)

By The JAX authors

© Copyright 2024, The JAX Authors.\
