- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.BlockSpec

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.BlockSpec.rst "Download source file")
-  .pdf

# jax.experimental.pallas.BlockSpec

## Contents

- [`BlockSpec`](#jax.experimental.pallas.BlockSpec)
  - [`BlockSpec.__init__()`](#jax.experimental.pallas.BlockSpec.__init__)

# jax.experimental.pallas.BlockSpec[\#](#jax-experimental-pallas-blockspec "Link to this heading")

*class* jax.experimental.pallas.BlockSpec(*block_shape=None*, *index_map=None*, *pipeline_mode=None*, *\**, *memory_space=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/core.py#L544-L711)[\#](#jax.experimental.pallas.BlockSpec "Link to this definition")  
Specifies how an array should be sliced for each invocation of a kernel.

The block_shape is a sequence of int \| None\`s, or \`BlockDim types (e.g. pl.Element, pl.Squeezed, pl.Blocked, pl.BoundedSlice). Each of these types specify the size of the block dimension. None is used to specify a dimension that is squeezed out of the kernel. The BlockDim types allow for more fine-grained control over the indexing of the dimension. The index_map needs to return a tuple of the same length as block_shape, which each entry depending on the type of BlockDim.

See [BlockSpec, a.k.a. how to chunk up inputs](../pallas/grid_blockspec.html#pallas-blockspec) and the individual BlockDim type docstrings for more details.

Parameters:  
- **block_shape** (*Sequence\[BlockDim* *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None\]* *\|* *None*)

- **index_map** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]* *\|* *None*)

- **pipeline_mode** (*Buffered* *\|* *None*)

- **memory_space** (*Any* *\|* *None*)

\_\_init\_\_(*block_shape=None*, *index_map=None*, *pipeline_mode=None*, *\**, *memory_space=None*)[\#](#jax.experimental.pallas.BlockSpec.__init__ "Link to this definition")  
Parameters:  
- **block_shape** (*Sequence\[BlockDim* *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None\]* *\|* *None*)

- **index_map** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]* *\|* *None*)

- **pipeline_mode** (*Buffered* *\|* *None*)

- **memory_space** (*Any* *\|* *None*)

Return type:  
None

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.pallas.BlockSpec.__init__ "jax.experimental.pallas.BlockSpec.__init__")(\[block_shape, index_map, ...\]) |  |
| `replace`(\*\*changes) | Return a new object replacing specified fields with new values. |
| `to_block_mapping`(origin, array_aval, \*, ...) |  |

Attributes

|                 |     |
|-----------------|-----|
| `block_shape`   |     |
| `index_map`     |     |
| `memory_space`  |     |
| `pipeline_mode` |     |

[](jax.experimental.pallas.triton.store.html "previous page")

previous

jax.experimental.pallas.triton.store

[](jax.experimental.pallas.GridSpec.html "next page")

next

jax.experimental.pallas.GridSpec

Contents

- [`BlockSpec`](#jax.experimental.pallas.BlockSpec)
  - [`BlockSpec.__init__()`](#jax.experimental.pallas.BlockSpec.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
