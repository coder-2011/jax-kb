- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.triton` module](../jax.experimental.pallas.triton.html)
- jax.experimental.pallas.triton.load

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.triton.load.rst "Download source file")
-  .pdf

# jax.experimental.pallas.triton.load

## Contents

- [`load()`](#jax.experimental.pallas.triton.load)

# jax.experimental.pallas.triton.load[\#](#jax-experimental-pallas-triton-load "Link to this heading")

jax.experimental.pallas.triton.load(*ref*, *\**, *mask=None*, *other=None*, *cache_modifier=None*, *eviction_policy=None*, *volatile=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/triton/primitives.py#L244-L277)[\#](#jax.experimental.pallas.triton.load "Link to this definition")  
Loads an array from the given ref.

If neither `mask` nor `other` is specified, this function has the same semantics as `ref[idx]` in JAX.

Parameters:  
- **ref** ([*Ref*](jax.ref.Ref.html#jax.ref.Ref "jax.ref.Ref")) – The ref to load from.

- **mask** ([*jax.Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*) – An optional boolean mask specifying which indices to load. If mask is `False` and `other` is not given, no assumptions can be made about the value in the resulting array.

- **other** (*jax.typing.ArrayLike* *\|* *None*) – An optional value to use for indices where mask is `False`.

- **cache_modifier** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – TO BE DOCUMENTED.

- **eviction_policy** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – TO BE DOCUMENTED.

- **volatile** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – TO BE DOCUMENTED.

Return type:  
[jax.Array](jax.Array.html#jax.Array "jax.Array")

[](jax.experimental.pallas.triton.elementwise_inline_asm.html "previous page")

previous

jax.experimental.pallas.triton.elementwise_inline_asm

[](jax.experimental.pallas.triton.max_contiguous.html "next page")

next

jax.experimental.pallas.triton.max_contiguous

Contents

- [`load()`](#jax.experimental.pallas.triton.load)

By The JAX authors

© Copyright 2024, The JAX Authors.\
