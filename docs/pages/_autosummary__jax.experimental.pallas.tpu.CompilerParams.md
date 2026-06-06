- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.CompilerParams

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.CompilerParams.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.CompilerParams

## Contents

- [`CompilerParams`](#jax.experimental.pallas.tpu.CompilerParams)
  - [`CompilerParams.dimension_semantics`](#jax.experimental.pallas.tpu.CompilerParams.dimension_semantics)
  - [`CompilerParams.allow_input_fusion`](#jax.experimental.pallas.tpu.CompilerParams.allow_input_fusion)
  - [`CompilerParams.vmem_limit_bytes`](#jax.experimental.pallas.tpu.CompilerParams.vmem_limit_bytes)
  - [`CompilerParams.collective_id`](#jax.experimental.pallas.tpu.CompilerParams.collective_id)
  - [`CompilerParams.has_side_effects`](#jax.experimental.pallas.tpu.CompilerParams.has_side_effects)
  - [`CompilerParams.flags`](#jax.experimental.pallas.tpu.CompilerParams.flags)
  - [`CompilerParams.internal_scratch_in_bytes`](#jax.experimental.pallas.tpu.CompilerParams.internal_scratch_in_bytes)
  - [`CompilerParams.serialization_format`](#jax.experimental.pallas.tpu.CompilerParams.serialization_format)
  - [`CompilerParams.disable_bounds_checks`](#jax.experimental.pallas.tpu.CompilerParams.disable_bounds_checks)
  - [`CompilerParams.disable_semaphore_checks`](#jax.experimental.pallas.tpu.CompilerParams.disable_semaphore_checks)
  - [`CompilerParams.skip_device_barrier`](#jax.experimental.pallas.tpu.CompilerParams.skip_device_barrier)
  - [`CompilerParams.allow_collective_id_without_custom_barrier`](#jax.experimental.pallas.tpu.CompilerParams.allow_collective_id_without_custom_barrier)
  - [`CompilerParams.use_tc_tiling_on_sc`](#jax.experimental.pallas.tpu.CompilerParams.use_tc_tiling_on_sc)
  - [`CompilerParams.needs_layout_passes`](#jax.experimental.pallas.tpu.CompilerParams.needs_layout_passes)
  - [`CompilerParams.fuse_transposed_lhs_in_matmul`](#jax.experimental.pallas.tpu.CompilerParams.fuse_transposed_lhs_in_matmul)
  - [`CompilerParams.__init__()`](#jax.experimental.pallas.tpu.CompilerParams.__init__)

# jax.experimental.pallas.tpu.CompilerParams[\#](#jax-experimental-pallas-tpu-compilerparams "Link to this heading")

*class* jax.experimental.pallas.tpu.CompilerParams(*dimension_semantics=None*, *allow_input_fusion=None*, *vmem_limit_bytes=None*, *collective_id=None*, *has_side_effects=False*, *flags=None*, *internal_scratch_in_bytes=None*, *serialization_format=1*, *disable_bounds_checks=False*, *disable_semaphore_checks=False*, *skip_device_barrier=False*, *allow_collective_id_without_custom_barrier=False*, *shape_invariant_numerics=True*, *use_tc_tiling_on_sc=None*, *needs_layout_passes=True*, *fuse_transposed_lhs_in_matmul=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/core.py#L86-L206)[\#](#jax.experimental.pallas.tpu.CompilerParams "Link to this definition")  
Mosaic TPU compiler parameters.

Parameters:  
- **dimension_semantics** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[DimensionSemantics,* *...\]* *\|* *None*)

- **allow_input_fusion** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *...\]* *\|* *None*)

- **vmem_limit_bytes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

- **collective_id** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

- **has_side_effects** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *SideEffectType*)

- **flags** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *Any\]* *\|* *None*)

- **internal_scratch_in_bytes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

- **serialization_format** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **disable_bounds_checks** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **disable_semaphore_checks** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **skip_device_barrier** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **allow_collective_id_without_custom_barrier** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **shape_invariant_numerics** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **use_tc_tiling_on_sc** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None*)

- **needs_layout_passes** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **fuse_transposed_lhs_in_matmul** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

dimension_semantics[\#](#jax.experimental.pallas.tpu.CompilerParams.dimension_semantics "Link to this definition")  
A list of dimension semantics for each grid dimension of the kernel. Either “parallel” for dimensions that can execute in any order, or “arbitrary” for dimensions that must be executed sequentially.

Type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[DimensionSemantics, …\] \| None

allow_input_fusion[\#](#jax.experimental.pallas.tpu.CompilerParams.allow_input_fusion "Link to this definition")  
A list of booleans indicating whether input fusion is allowed for each argument.

Type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"), …\] \| None

vmem_limit_bytes[\#](#jax.experimental.pallas.tpu.CompilerParams.vmem_limit_bytes "Link to this definition")  
Overrides the default VMEM limit for a kernel. Note that this must be used in conjunction with the –xla_tpu_scoped_vmem_limit_kib=N flag with N\*1kib \> vmem_limit_bytes.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| None

collective_id[\#](#jax.experimental.pallas.tpu.CompilerParams.collective_id "Link to this definition")  
Indicates which barrier semaphore to use for the kernel. Note that using the same collective_id does not guarantee that the same barrier semaphore will be allocated between kernels.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| None

has_side_effects[\#](#jax.experimental.pallas.tpu.CompilerParams.has_side_effects "Link to this definition")  
Set to True to prevent kernel being CSEd by XLA.

Type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") \| SideEffectType

flags[\#](#jax.experimental.pallas.tpu.CompilerParams.flags "Link to this definition")  
A dictionary of command line flags for the kernel.

Type:  
[dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"), Any\] \| None

internal_scratch_in_bytes[\#](#jax.experimental.pallas.tpu.CompilerParams.internal_scratch_in_bytes "Link to this definition")  
The size of the internal scratch space used by Mosaic.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| None

serialization_format[\#](#jax.experimental.pallas.tpu.CompilerParams.serialization_format "Link to this definition")  
The serialization format for the kernel body.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

disable_bounds_checks[\#](#jax.experimental.pallas.tpu.CompilerParams.disable_bounds_checks "Link to this definition")  
Disable bounds checks in the kernel.

Type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

disable_semaphore_checks[\#](#jax.experimental.pallas.tpu.CompilerParams.disable_semaphore_checks "Link to this definition")  
Disable semaphore checks in the kernel.

Type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

skip_device_barrier[\#](#jax.experimental.pallas.tpu.CompilerParams.skip_device_barrier "Link to this definition")  
Skip the default device barrier for the kernel.

Type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

allow_collective_id_without_custom_barrier[\#](#jax.experimental.pallas.tpu.CompilerParams.allow_collective_id_without_custom_barrier "Link to this definition")  
Allow the use of collective_id without a custom barrier.

Type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

use_tc_tiling_on_sc[\#](#jax.experimental.pallas.tpu.CompilerParams.use_tc_tiling_on_sc "Link to this definition")  
Use TensorCore tiling for SparseCore. This flag is only used for `SC_*_SUBCORE` kernels and it implicitly defaults to True.

Type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") \| None

needs_layout_passes[\#](#jax.experimental.pallas.tpu.CompilerParams.needs_layout_passes "Link to this definition")  
Whether to use vector layout inference passes. This flag is temporary and will eventually be removed.

Type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

fuse_transposed_lhs_in_matmul[\#](#jax.experimental.pallas.tpu.CompilerParams.fuse_transposed_lhs_in_matmul "Link to this definition")  
Hint to compilers to attempt to fuse transposed LHS in MXU if users specify the transposed layout of LHS in matmul operations, e.g., jnp.einsum(‘km,kn-\>mn’, lhs, rhs); on the other hand, When transposition is performed separately from multiplication (e.g. jnp.matmul(lhs.T, rhs)), this flag does not affect the compiler’s decision (it might still decide to do it if obviously profitable). Note that this flag is at the best-effort basis, and the fusion will only be performed when compilers determine it is feasible. Also, the fusion is not always profitable and therefore should be used sparingly.

Type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

\_\_init\_\_(*dimension_semantics=None*, *allow_input_fusion=None*, *vmem_limit_bytes=None*, *collective_id=None*, *has_side_effects=False*, *flags=None*, *internal_scratch_in_bytes=None*, *serialization_format=1*, *disable_bounds_checks=False*, *disable_semaphore_checks=False*, *skip_device_barrier=False*, *allow_collective_id_without_custom_barrier=False*, *shape_invariant_numerics=True*, *use_tc_tiling_on_sc=None*, *needs_layout_passes=True*, *fuse_transposed_lhs_in_matmul=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/core.py#L144-L203)[\#](#jax.experimental.pallas.tpu.CompilerParams.__init__ "Link to this definition")  
Parameters:  
- **dimension_semantics** (*Sequence\[DimensionSemantics\]* *\|* *None*)

- **allow_input_fusion** (*Sequence\[*[*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]* *\|* *None*)

- **vmem_limit_bytes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

- **collective_id** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

- **has_side_effects** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *SideEffectType*)

- **flags** (*Mapping\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *Any\]* *\|* *None*)

- **internal_scratch_in_bytes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

- **serialization_format** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **disable_bounds_checks** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **disable_semaphore_checks** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **skip_device_barrier** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **allow_collective_id_without_custom_barrier** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **shape_invariant_numerics** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **use_tc_tiling_on_sc** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None*)

- **needs_layout_passes** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **fuse_transposed_lhs_in_matmul** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.pallas.tpu.CompilerParams.__init__ "jax.experimental.pallas.tpu.CompilerParams.__init__")(\[dimension_semantics, ...\]) |  |
| `replace`(\*\*changes) | Return a new object replacing specified fields with new values. |

Attributes

|  |  |
|----|----|
| [`allow_collective_id_without_custom_barrier`](#jax.experimental.pallas.tpu.CompilerParams.allow_collective_id_without_custom_barrier "jax.experimental.pallas.tpu.CompilerParams.allow_collective_id_without_custom_barrier") |  |
| [`allow_input_fusion`](#jax.experimental.pallas.tpu.CompilerParams.allow_input_fusion "jax.experimental.pallas.tpu.CompilerParams.allow_input_fusion") |  |
| [`collective_id`](#jax.experimental.pallas.tpu.CompilerParams.collective_id "jax.experimental.pallas.tpu.CompilerParams.collective_id") |  |
| [`dimension_semantics`](#jax.experimental.pallas.tpu.CompilerParams.dimension_semantics "jax.experimental.pallas.tpu.CompilerParams.dimension_semantics") |  |
| [`disable_bounds_checks`](#jax.experimental.pallas.tpu.CompilerParams.disable_bounds_checks "jax.experimental.pallas.tpu.CompilerParams.disable_bounds_checks") |  |
| [`disable_semaphore_checks`](#jax.experimental.pallas.tpu.CompilerParams.disable_semaphore_checks "jax.experimental.pallas.tpu.CompilerParams.disable_semaphore_checks") |  |
| [`flags`](#jax.experimental.pallas.tpu.CompilerParams.flags "jax.experimental.pallas.tpu.CompilerParams.flags") |  |
| [`fuse_transposed_lhs_in_matmul`](#jax.experimental.pallas.tpu.CompilerParams.fuse_transposed_lhs_in_matmul "jax.experimental.pallas.tpu.CompilerParams.fuse_transposed_lhs_in_matmul") |  |
| [`has_side_effects`](#jax.experimental.pallas.tpu.CompilerParams.has_side_effects "jax.experimental.pallas.tpu.CompilerParams.has_side_effects") |  |
| [`internal_scratch_in_bytes`](#jax.experimental.pallas.tpu.CompilerParams.internal_scratch_in_bytes "jax.experimental.pallas.tpu.CompilerParams.internal_scratch_in_bytes") |  |
| [`needs_layout_passes`](#jax.experimental.pallas.tpu.CompilerParams.needs_layout_passes "jax.experimental.pallas.tpu.CompilerParams.needs_layout_passes") |  |
| [`serialization_format`](#jax.experimental.pallas.tpu.CompilerParams.serialization_format "jax.experimental.pallas.tpu.CompilerParams.serialization_format") |  |
| `shape_invariant_numerics` |  |
| [`skip_device_barrier`](#jax.experimental.pallas.tpu.CompilerParams.skip_device_barrier "jax.experimental.pallas.tpu.CompilerParams.skip_device_barrier") |  |
| [`use_tc_tiling_on_sc`](#jax.experimental.pallas.tpu.CompilerParams.use_tc_tiling_on_sc "jax.experimental.pallas.tpu.CompilerParams.use_tc_tiling_on_sc") |  |
| [`vmem_limit_bytes`](#jax.experimental.pallas.tpu.CompilerParams.vmem_limit_bytes "jax.experimental.pallas.tpu.CompilerParams.vmem_limit_bytes") |  |

[](jax.experimental.pallas.tpu.ChipVersion.html "previous page")

previous

jax.experimental.pallas.tpu.ChipVersion

[](jax.experimental.pallas.tpu.GridDimensionSemantics.html "next page")

next

jax.experimental.pallas.tpu.GridDimensionSemantics

Contents

- [`CompilerParams`](#jax.experimental.pallas.tpu.CompilerParams)
  - [`CompilerParams.dimension_semantics`](#jax.experimental.pallas.tpu.CompilerParams.dimension_semantics)
  - [`CompilerParams.allow_input_fusion`](#jax.experimental.pallas.tpu.CompilerParams.allow_input_fusion)
  - [`CompilerParams.vmem_limit_bytes`](#jax.experimental.pallas.tpu.CompilerParams.vmem_limit_bytes)
  - [`CompilerParams.collective_id`](#jax.experimental.pallas.tpu.CompilerParams.collective_id)
  - [`CompilerParams.has_side_effects`](#jax.experimental.pallas.tpu.CompilerParams.has_side_effects)
  - [`CompilerParams.flags`](#jax.experimental.pallas.tpu.CompilerParams.flags)
  - [`CompilerParams.internal_scratch_in_bytes`](#jax.experimental.pallas.tpu.CompilerParams.internal_scratch_in_bytes)
  - [`CompilerParams.serialization_format`](#jax.experimental.pallas.tpu.CompilerParams.serialization_format)
  - [`CompilerParams.disable_bounds_checks`](#jax.experimental.pallas.tpu.CompilerParams.disable_bounds_checks)
  - [`CompilerParams.disable_semaphore_checks`](#jax.experimental.pallas.tpu.CompilerParams.disable_semaphore_checks)
  - [`CompilerParams.skip_device_barrier`](#jax.experimental.pallas.tpu.CompilerParams.skip_device_barrier)
  - [`CompilerParams.allow_collective_id_without_custom_barrier`](#jax.experimental.pallas.tpu.CompilerParams.allow_collective_id_without_custom_barrier)
  - [`CompilerParams.use_tc_tiling_on_sc`](#jax.experimental.pallas.tpu.CompilerParams.use_tc_tiling_on_sc)
  - [`CompilerParams.needs_layout_passes`](#jax.experimental.pallas.tpu.CompilerParams.needs_layout_passes)
  - [`CompilerParams.fuse_transposed_lhs_in_matmul`](#jax.experimental.pallas.tpu.CompilerParams.fuse_transposed_lhs_in_matmul)
  - [`CompilerParams.__init__()`](#jax.experimental.pallas.tpu.CompilerParams.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
