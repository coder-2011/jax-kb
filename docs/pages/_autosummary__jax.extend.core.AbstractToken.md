- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.core` module](../jax.extend.core.html)
- jax.extend.core.AbstractToken

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.core.AbstractToken.rst "Download source file")
-  .pdf

# jax.extend.core.AbstractToken

## Contents

- [`AbstractToken`](#jax.extend.core.AbstractToken)
  - [`AbstractToken.__init__()`](#jax.extend.core.AbstractToken.__init__)

# jax.extend.core.AbstractToken[\#](#jax-extend-core-abstracttoken "Link to this heading")

*class* jax.extend.core.AbstractToken[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py#L2954-L2958)[\#](#jax.extend.core.AbstractToken "Link to this definition")  
\_\_init\_\_()[\#](#jax.extend.core.AbstractToken.__init__ "Link to this definition")  

Methods

|  |  |
|----|----|
| [`__init__`](#jax.extend.core.AbstractToken.__init__ "jax.extend.core.AbstractToken.__init__")() |  |
| `at_least_vspace`() |  |
| `dec_rank`(size, spec) |  |
| `inc_rank`(size, spec) |  |
| `leading_axis_spec`() |  |
| `lo_ty`() |  |
| `lo_ty_qdd`(qdd) |  |
| `lower_val2`(hi_val) |  |
| `normalize`() |  |
| `raise_val2`(lo_vals_ft) |  |
| `shard`(mesh, manual_axes, check_vma, spec) |  |
| `str_short`(\[short_dtypes, mesh_axis_types\]) |  |
| `strip_weak_type`() |  |
| `to_ct_aval`() |  |
| `to_tangent_aval`() |  |
| `unshard`(mesh, check_vma, spec) |  |
| `update`(\*\*kwargs) |  |
| `update_manual_axis_type`(mat) |  |
| `update_weak_type`(weak_type) |  |
| `vspace_add`(x, y) |  |

Attributes

|           |     |
|-----------|-----|
| `has_qdd` |     |
| `is_high` |     |

[](../jax.extend.core.html "previous page")

previous

`jax.extend.core` module

[](jax.extend.core.CallPrimitive.html "next page")

next

jax.extend.core.CallPrimitive

Contents

- [`AbstractToken`](#jax.extend.core.AbstractToken)
  - [`AbstractToken.__init__()`](#jax.extend.core.AbstractToken.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
