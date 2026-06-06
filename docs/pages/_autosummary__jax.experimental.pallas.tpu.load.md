- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.load

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.load.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.load

## Contents

- [`load()`](#jax.experimental.pallas.tpu.load)

# jax.experimental.pallas.tpu.load[\#](#jax-experimental-pallas-tpu-load "Link to this heading")

jax.experimental.pallas.tpu.load(*ref*, *\**, *mask=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/primitives.py#L1060-L1074)[\#](#jax.experimental.pallas.tpu.load "Link to this definition")  
Loads an array from the given ref.

If `mask` is not specified, this function has the same semantics as `ref[idx]` in JAX.

Parameters:  
- **ref** ([*Ref*](jax.ref.Ref.html#jax.ref.Ref "jax.ref.Ref")) – The ref to load from.

- **mask** ([*jax.Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*) – An optional boolean mask specifying which indices to load.

Returns:  
The loaded array.

Return type:  
[jax.Array](jax.Array.html#jax.Array "jax.Array")

[](jax.experimental.pallas.tpu.TpuInfo.html "previous page")

previous

jax.experimental.pallas.tpu.TpuInfo

[](jax.experimental.pallas.tpu.store.html "next page")

next

jax.experimental.pallas.tpu.store

Contents

- [`load()`](#jax.experimental.pallas.tpu.load)

By The JAX authors

© Copyright 2024, The JAX Authors.\
