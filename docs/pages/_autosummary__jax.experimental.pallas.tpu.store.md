- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.store

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.store.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.store

## Contents

- [`store()`](#jax.experimental.pallas.tpu.store)

# jax.experimental.pallas.tpu.store[\#](#jax-experimental-pallas-tpu-store "Link to this heading")

jax.experimental.pallas.tpu.store(*ref*, *val*, *\**, *mask=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/primitives.py#L1076-L1088)[\#](#jax.experimental.pallas.tpu.store "Link to this definition")  
Stores a value to the given ref.

If `mask` is not specified, this function has the same semantics as `ref[idx]`` ``=`` ``val` in JAX.

Parameters:  
- **ref** ([*Ref*](jax.ref.Ref.html#jax.ref.Ref "jax.ref.Ref")) – The ref to store to.

- **val** ([*jax.Array*](jax.Array.html#jax.Array "jax.Array")) – The value to store.

- **mask** ([*jax.Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*) – An optional boolean mask specifying which indices to store.

Return type:  
None

[](jax.experimental.pallas.tpu.load.html "previous page")

previous

jax.experimental.pallas.tpu.load

[](jax.experimental.pallas.tpu.async_copy.html "next page")

next

jax.experimental.pallas.tpu.async_copy

Contents

- [`store()`](#jax.experimental.pallas.tpu.store)

By The JAX authors

© Copyright 2024, The JAX Authors.\
