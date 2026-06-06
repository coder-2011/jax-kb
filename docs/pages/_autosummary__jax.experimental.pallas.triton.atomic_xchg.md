- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.triton` module](../jax.experimental.pallas.triton.html)
- jax.experimental.pallas.triton.atomic_xchg

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.triton.atomic_xchg.rst "Download source file")
-  .pdf

# jax.experimental.pallas.triton.atomic_xchg

## Contents

- [`atomic_xchg()`](#jax.experimental.pallas.triton.atomic_xchg)

# jax.experimental.pallas.triton.atomic_xchg[\#](#jax-experimental-pallas-triton-atomic-xchg "Link to this heading")

jax.experimental.pallas.triton.atomic_xchg(*x_ref_or_view*, *idx*, *val*, *\**, *mask=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/triton/primitives.py#L504-L518)[\#](#jax.experimental.pallas.triton.atomic_xchg "Link to this definition")  
Atomically exchanges the given value with the value at the given index.

Parameters:  
- **x_ref_or_view** – The ref to operate on.

- **idx** – The indexer to use.

- **mask** (*Any* *\|* *None*) – TO BE DOCUMENTED.

Returns:  
The value at the given index prior to the aupdate.

[](jax.experimental.pallas.triton.atomic_or.html "previous page")

previous

jax.experimental.pallas.triton.atomic_or

[](jax.experimental.pallas.triton.atomic_xor.html "next page")

next

jax.experimental.pallas.triton.atomic_xor

Contents

- [`atomic_xchg()`](#jax.experimental.pallas.triton.atomic_xchg)

By The JAX authors

© Copyright 2024, The JAX Authors.\
