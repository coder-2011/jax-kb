- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.pallas` module](../jax.extend.pallas.html)
- jax.extend.pallas.GridMapping

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.pallas.GridMapping.rst "Download source file")
-  .pdf

# jax.extend.pallas.GridMapping

## Contents

- [`GridMapping`](#jax.extend.pallas.GridMapping)
  - [`GridMapping.__init__()`](#jax.extend.pallas.GridMapping.__init__)

# jax.extend.pallas.GridMapping[\#](#jax-extend-pallas-gridmapping "Link to this heading")

*class* jax.extend.pallas.GridMapping(*grid*, *grid_names*, *block_mappings*, *index_map_tree*, *index_map_avals*, *vmapped_dims*, *scratch_avals*, *num_index_operands*, *num_inputs*, *num_outputs*, *get_grid_indices=None*, *local_grid_env=None*, *debug=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/core.py#L924-L1144)[\#](#jax.extend.pallas.GridMapping "Link to this definition")  
An internal canonicalized version of GridSpec.

Encodes the calling conventions of the pallas_call primitive, the kernel, and the index maps.

The pallas_call is invoked with: `*dynamic_grid_sizes,`` ``*index,`` ``*inputs`. The `index` operands are for the scalar prefetch.

The kernel function is invoked with: `*index,`` ``*inputs,`` ``*scratch`.

The index map functions are invoked with: `*program_ids,`` ``*index`.

See the check_invariants method for a more precise specification.

Parameters:  
- **grid** (*GridMappingGrid*)

- **grid_names** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Hashable,* *...\]* *\|* *None*)

- **block_mappings** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[BlockMapping,* *...\]*)

- **index_map_tree** (*tree_util.PyTreeDef*)

- **index_map_avals** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[jax_core.AbstractValue,* *...\]*)

- **vmapped_dims** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*)

- **scratch_avals** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[jax_core.AbstractValue,* *...\]*)

- **num_index_operands** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **num_inputs** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **num_outputs** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **get_grid_indices** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable") *\|* *None*)

- **local_grid_env** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable") *\|* *None*)

- **debug** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

\_\_init\_\_(*grid*, *grid_names*, *block_mappings*, *index_map_tree*, *index_map_avals*, *vmapped_dims*, *scratch_avals*, *num_index_operands*, *num_inputs*, *num_outputs*, *get_grid_indices=None*, *local_grid_env=None*, *debug=False*)[\#](#jax.extend.pallas.GridMapping.__init__ "Link to this definition")  
Parameters:  
- **grid** (*GridMappingGrid*)

- **grid_names** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Hashable,* *...\]* *\|* *None*)

- **block_mappings** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[BlockMapping,* *...\]*)

- **index_map_tree** (*tree_util.PyTreeDef*)

- **index_map_avals** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[jax_core.AbstractValue,* *...\]*)

- **vmapped_dims** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*)

- **scratch_avals** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[jax_core.AbstractValue,* *...\]*)

- **num_index_operands** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **num_inputs** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **num_outputs** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **get_grid_indices** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable") *\|* *None*)

- **local_grid_env** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable") *\|* *None*)

- **debug** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Return type:  
None

Methods

|  |  |
|----|----|
| [`__init__`](#jax.extend.pallas.GridMapping.__init__ "jax.extend.pallas.GridMapping.__init__")(grid, grid_names, block_mappings, ...) |  |
| `check_invariants`() |  |
| `replace`(\*\*kwargs) |  |
| `to_lojax`() |  |
| `trace_env`() |  |

Attributes

|  |  |
|----|----|
| `block_mappings_output` |  |
| `debug` |  |
| `get_grid_indices` |  |
| `in_shapes` | The shapes of `*index`, `*inputs`. |
| `local_grid_env` |  |
| `num_dynamic_grid_bounds` |  |
| `num_scratch_operands` |  |
| `out_shapes` |  |
| `slice_block_ops` | Returns a slice to select the block operands to a kernel. |
| `slice_index_ops` | Returns a slice object to select the index operands to a kernel. |
| `slice_scratch_ops` | Returns a slice object to select the scratch operands to a kernel. |
| `static_grid` |  |
| `grid` |  |
| `grid_names` |  |
| `block_mappings` |  |
| `index_map_tree` |  |
| `index_map_avals` |  |
| `vmapped_dims` |  |
| `scratch_avals` |  |
| `num_index_operands` |  |
| `num_inputs` |  |
| `num_outputs` |  |

[](../jax.extend.pallas.html "previous page")

previous

`jax.extend.pallas` module

[](jax.extend.pallas.register_lowering_rule.html "next page")

next

jax.extend.pallas.register_lowering_rule

Contents

- [`GridMapping`](#jax.extend.pallas.GridMapping)
  - [`GridMapping.__init__()`](#jax.extend.pallas.GridMapping.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
