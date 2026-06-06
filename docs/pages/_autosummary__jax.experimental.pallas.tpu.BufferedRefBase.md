- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.BufferedRefBase

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.BufferedRefBase.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.BufferedRefBase

## Contents

- [`BufferedRefBase`](#jax.experimental.pallas.tpu.BufferedRefBase)
  - [`BufferedRefBase.__init__()`](#jax.experimental.pallas.tpu.BufferedRefBase.__init__)

# jax.experimental.pallas.tpu.BufferedRefBase[\#](#jax-experimental-pallas-tpu-bufferedrefbase "Link to this heading")

*class* jax.experimental.pallas.tpu.BufferedRefBase[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/pipeline.py#L259-L425)[\#](#jax.experimental.pallas.tpu.BufferedRefBase "Link to this definition")  
Abstract interface for BufferedRefs.

\_\_init\_\_()[\#](#jax.experimental.pallas.tpu.BufferedRefBase.__init__ "Link to this definition")  

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.pallas.tpu.BufferedRefBase.__init__ "jax.experimental.pallas.tpu.BufferedRefBase.__init__")() |  |
| `advance_copy_in_slot`(\[predicate\]) | Advance the copy in slot. |
| `advance_copy_out_slot`(\[predicate\]) | Advance the copy out slot. |
| `advance_wait_in_slot`(\[predicate\]) | Advance the wait in slot. |
| `advance_wait_out_slot`(\[predicate\]) | Advance the wait out slot. |
| `bind_existing_ref`(window_ref, indices) | For handling VMEM references, the pipeline aliases the existing ref. |
| `get_dma_slice`(src_ty, grid_indices) |  |
| `initialize_slots`() | Initializes slots to 0. |
| `unbind_refs`() |  |
| `with_spec`(spec) | Returns a new BufferedRefBase with the given block spec. |

Attributes

|  |  |
|----|----|
| `block_shape` |  |
| `buffer_type` |  |
| `compute_index` |  |
| `has_allocated_buffer` | Returns True if the reference has an allocated buffer outside loop. |
| `has_indirect` | Whether any block dimension uses indirect indexing. |
| `is_buffered` |  |
| `is_input` |  |
| `is_input_output` |  |
| `is_manual` |  |
| `is_output` |  |
| `is_trivial_windowing` | Whether the reference uses trivial windowing. |
| `spec` |  |

[](jax.experimental.pallas.tpu.BufferedRef.html "previous page")

previous

jax.experimental.pallas.tpu.BufferedRef

[](jax.experimental.pallas.tpu.emit_pipeline.html "next page")

next

jax.experimental.pallas.tpu.emit_pipeline

Contents

- [`BufferedRefBase`](#jax.experimental.pallas.tpu.BufferedRefBase)
  - [`BufferedRefBase.__init__()`](#jax.experimental.pallas.tpu.BufferedRefBase.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
