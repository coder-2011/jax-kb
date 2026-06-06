- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.triton` module](../jax.experimental.pallas.triton.html)
- jax.experimental.pallas.triton.atomic_cas

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.triton.atomic_cas.rst "Download source file")
-  .pdf

# jax.experimental.pallas.triton.atomic_cas

## Contents

- [`atomic_cas()`](#jax.experimental.pallas.triton.atomic_cas)

# jax.experimental.pallas.triton.atomic_cas[\#](#jax-experimental-pallas-triton-atomic-cas "Link to this heading")

jax.experimental.pallas.triton.atomic_cas(*ref*, *cmp*, *val*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/triton/primitives.py#L634-L648)[\#](#jax.experimental.pallas.triton.atomic_cas "Link to this definition")  
Performs an atomic compare-and-swap of the value in the ref with the

given value.

Parameters:  
- **ref** – The ref to operate on.

- **cmp** – The expected value to compare against.

- **val** – The value to swap in.

Returns:  
The value at the given index prior to the atomic operation.

[](jax.experimental.pallas.triton.atomic_add.html "previous page")

previous

jax.experimental.pallas.triton.atomic_add

[](jax.experimental.pallas.triton.atomic_max.html "next page")

next

jax.experimental.pallas.triton.atomic_max

Contents

- [`atomic_cas()`](#jax.experimental.pallas.triton.atomic_cas)

By The JAX authors

© Copyright 2024, The JAX Authors.\
