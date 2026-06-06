- [](../index.html)
- [API Reference](../jax.html)
- [`jax.ref` module](../jax.ref.html)
- jax.ref.AbstractRef

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.ref.AbstractRef.rst "Download source file")
-  .pdf

# jax.ref.AbstractRef

## Contents

- [`AbstractRef`](#jax.ref.AbstractRef)
  - [`AbstractRef.__init__()`](#jax.ref.AbstractRef.__init__)

# jax.ref.AbstractRef[\#](#jax-ref-abstractref "Link to this heading")

*class* jax.ref.AbstractRef(*inner_aval*, *memory_space=None*, *kind=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/state/types.py#L434-L633)[\#](#jax.ref.AbstractRef "Link to this definition")  
Abstract mutable array reference.

Refer to the [Ref guide](https://docs.jax.dev/en/latest/array_refs.html) for more information.

Parameters:  
- **inner_aval** (*core.AbstractValue*)

- **memory_space** (*Any*)

- **kind** (*Any*)

\_\_init\_\_(*inner_aval*, *memory_space=None*, *kind=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/state/types.py#L443-L461)[\#](#jax.ref.AbstractRef.__init__ "Link to this definition")  
Parameters:  
- **inner_aval** (*core.AbstractValue*)

- **memory_space** (*Any*)

- **kind** (*Any*)

Methods

|  |  |
|----|----|
| [`__init__`](#jax.ref.AbstractRef.__init__ "jax.ref.AbstractRef.__init__")(inner_aval\[, memory_space, kind\]) |  |
| `at_least_vspace`() |  |
| `dec_rank`(size, spec) |  |
| `inc_rank`(size, spec) |  |
| `leading_axis_spec`() |  |
| `lo_ty`() |  |
| `lo_ty_qdd`(qdd) |  |
| `lower_val`(ref) |  |
| `lower_val2`(hi_val) |  |
| `normalize`() |  |
| `raise_val`(\*vals) |  |
| `raise_val2`(lo_vals_ft) |  |
| `shard`(mesh, manual_axes, check_vma, spec) |  |
| `str_short`(\[short_dtypes, mesh_axis_types\]) |  |
| `strip_weak_type`() |  |
| `to_ct_aval`() |  |
| `to_tangent_aval`() |  |
| `unshard`(mesh, check_vma, spec) |  |
| `update`(\[inner_aval, memory_space, kind\]) |  |
| `update_manual_axis_type`(mat) |  |
| `update_weak_type`(weak_type) |  |
| `vspace_add`(x, y) |  |

Attributes

|                    |     |
|--------------------|-----|
| `inner_aval`       |     |
| `memory_space`     |     |
| `kind`             |     |
| `T`                |     |
| `addupdate`        |     |
| `at`               |     |
| `bitcast`          |     |
| `dtype`            |     |
| `get`              |     |
| `has_qdd`          |     |
| `is_high`          |     |
| `manual_axis_type` |     |
| `mat`              |     |
| `ndim`             |     |
| `reshape`          |     |
| `set`              |     |
| `shape`            |     |
| `sharding`         |     |
| `size`             |     |
| `swap`             |     |
| `transpose`        |     |
| `weak_type`        |     |

[](../jax.ref.html "previous page")

previous

`jax.ref` module

[](jax.ref.Ref.html "next page")

next

jax.ref.Ref

Contents

- [`AbstractRef`](#jax.ref.AbstractRef)
  - [`AbstractRef.__init__()`](#jax.ref.AbstractRef.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
