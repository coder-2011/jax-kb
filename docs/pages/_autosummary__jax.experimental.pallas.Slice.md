- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.Slice

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.Slice.rst "Download source file")
-  .pdf

# jax.experimental.pallas.Slice

## Contents

- [`Slice`](#jax.experimental.pallas.Slice)
  - [`Slice.__init__()`](#jax.experimental.pallas.Slice.__init__)

# jax.experimental.pallas.Slice[\#](#jax-experimental-pallas-slice "Link to this heading")

*class* jax.experimental.pallas.Slice(*start*, *size*, *stride=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/indexing.py#L24-L73)[\#](#jax.experimental.pallas.Slice "Link to this definition")  
A slice with a start index and a size.

Both start index and size can either be static, i.e. known at tracing and compilation time, or dynamic.

Parameters:  
- **start** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*Array*](jax.Array.html#jax.Array "jax.Array"))

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*Array*](jax.Array.html#jax.Array "jax.Array"))

- **stride** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

\_\_init\_\_(*start*, *size*, *stride=1*)[\#](#jax.experimental.pallas.Slice.__init__ "Link to this definition")  
Parameters:  
- **start** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*Array*](jax.Array.html#jax.Array "jax.Array"))

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*Array*](jax.Array.html#jax.Array "jax.Array"))

- **stride** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

Return type:  
None

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.pallas.Slice.__init__ "jax.experimental.pallas.Slice.__init__")(start, size\[, stride\]) |  |
| `from_slice`(slc, size) |  |
| `tree_flatten`() |  |
| `tree_unflatten`(aux_data, children) |  |

Attributes

|                    |     |
|--------------------|-----|
| `start`            |     |
| `size`             |     |
| `stride`           |     |
| `is_dynamic_size`  |     |
| `is_dynamic_start` |     |

[](jax.experimental.pallas.GridSpec.html "previous page")

previous

jax.experimental.pallas.GridSpec

[](jax.experimental.pallas.core_map.html "next page")

next

jax.experimental.pallas.core_map

Contents

- [`Slice`](#jax.experimental.pallas.Slice)
  - [`Slice.__init__()`](#jax.experimental.pallas.Slice.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
