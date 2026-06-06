- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.triton` module](../jax.experimental.pallas.triton.html)
- jax.experimental.pallas.triton.store

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.triton.store.rst "Download source file")
-  .pdf

# jax.experimental.pallas.triton.store

## Contents

- [`store()`](#jax.experimental.pallas.triton.store)

# jax.experimental.pallas.triton.store[\#](#jax-experimental-pallas-triton-store "Link to this heading")

jax.experimental.pallas.triton.store(*ref*, *val*, *\**, *mask=None*, *eviction_policy=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/triton/primitives.py#L279-L297)[\#](#jax.experimental.pallas.triton.store "Link to this definition")  
Stores a value to the given ref.

See `load()` for the meaning of the arguments.

Parameters:  
- **ref** ([*Ref*](jax.ref.Ref.html#jax.ref.Ref "jax.ref.Ref"))

- **val** ([*jax.Array*](jax.Array.html#jax.Array "jax.Array"))

- **mask** ([*jax.Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*)

- **eviction_policy** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*)

Return type:  
None

[](jax.experimental.pallas.triton.max_contiguous.html "previous page")

previous

jax.experimental.pallas.triton.max_contiguous

[](jax.experimental.pallas.BlockSpec.html "next page")

next

jax.experimental.pallas.BlockSpec

Contents

- [`store()`](#jax.experimental.pallas.triton.store)

By The JAX authors

© Copyright 2024, The JAX Authors.\
