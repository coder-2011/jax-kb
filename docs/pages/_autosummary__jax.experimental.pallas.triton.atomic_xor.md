- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.triton` module](../jax.experimental.pallas.triton.html)
- jax.experimental.pallas.triton.atomic_xor

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.triton.atomic_xor.rst "Download source file")
-  .pdf

# jax.experimental.pallas.triton.atomic_xor

## Contents

- [`atomic_xor()`](#jax.experimental.pallas.triton.atomic_xor)

# jax.experimental.pallas.triton.atomic_xor[\#](#jax-experimental-pallas-triton-atomic-xor "Link to this heading")

jax.experimental.pallas.triton.atomic_xor(*x_ref_or_view*, *idx*, *val*, *\**, *mask=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/triton/primitives.py#L600-L614)[\#](#jax.experimental.pallas.triton.atomic_xor "Link to this definition")  
Atomically computes `x_ref_or_view[idx]`` ``^=`` ``val`.

Parameters:  
- **x_ref_or_view** – The ref to operate on.

- **idx** – The indexer to use.

- **mask** (*Any* *\|* *None*) – TO BE DOCUMENTED.

Returns:  
The value at the given index prior to the atomic operation.

[](jax.experimental.pallas.triton.atomic_xchg.html "previous page")

previous

jax.experimental.pallas.triton.atomic_xchg

[](jax.experimental.pallas.triton.approx_tanh.html "next page")

next

jax.experimental.pallas.triton.approx_tanh

Contents

- [`atomic_xor()`](#jax.experimental.pallas.triton.atomic_xor)

By The JAX authors

© Copyright 2024, The JAX Authors.\
