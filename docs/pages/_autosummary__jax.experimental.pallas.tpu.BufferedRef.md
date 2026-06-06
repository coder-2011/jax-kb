- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.BufferedRef

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.BufferedRef.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.BufferedRef

## Contents

- [`BufferedRef`](#jax.experimental.pallas.tpu.BufferedRef)
  - [`BufferedRef.spec`](#jax.experimental.pallas.tpu.BufferedRef.spec)
  - [`BufferedRef.buffer_type`](#jax.experimental.pallas.tpu.BufferedRef.buffer_type)
  - [`BufferedRef.window_ref`](#jax.experimental.pallas.tpu.BufferedRef.window_ref)
  - [`BufferedRef.copy_in_slot`](#jax.experimental.pallas.tpu.BufferedRef.copy_in_slot)
  - [`BufferedRef.copy_out_slot`](#jax.experimental.pallas.tpu.BufferedRef.copy_out_slot)
  - [`BufferedRef.wait_in_slot`](#jax.experimental.pallas.tpu.BufferedRef.wait_in_slot)
  - [`BufferedRef.wait_out_slot`](#jax.experimental.pallas.tpu.BufferedRef.wait_out_slot)
  - [`BufferedRef.next_fetch`](#jax.experimental.pallas.tpu.BufferedRef.next_fetch)
  - [`BufferedRef.sem_recvs`](#jax.experimental.pallas.tpu.BufferedRef.sem_recvs)
  - [`BufferedRef.sem_sends`](#jax.experimental.pallas.tpu.BufferedRef.sem_sends)
  - [`BufferedRef.tiling`](#jax.experimental.pallas.tpu.BufferedRef.tiling)
  - [`BufferedRef.is_trivial_windowing`](#jax.experimental.pallas.tpu.BufferedRef.is_trivial_windowing)
  - [`BufferedRef.has_allocated_buffer`](#jax.experimental.pallas.tpu.BufferedRef.has_allocated_buffer)
  - [`BufferedRef.__init__()`](#jax.experimental.pallas.tpu.BufferedRef.__init__)

# jax.experimental.pallas.tpu.BufferedRef[\#](#jax-experimental-pallas-tpu-bufferedref "Link to this heading")

*class* jax.experimental.pallas.tpu.BufferedRef(*\_spec*, *\_buffer_type*, *\_buffer_count*, *\_grid_rank*, *window_ref*, *copy_in_slot*, *wait_in_slot*, *copy_out_slot*, *wait_out_slot*, *next_fetch*, *sem_recvs*, *sem_sends*, *tiling*, *is_trivial_windowing=False*, *has_allocated_buffer=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/pipeline.py#L436-L914)[\#](#jax.experimental.pallas.tpu.BufferedRef "Link to this definition")  
A helper class to automate VMEM double buffering in pallas pipelines.

Parameters:  
- **\_spec** (*pallas_core.BlockSpec*)

- **\_buffer_type** (*BufferType*)

- **\_buffer_count** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **\_grid_rank** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

- **window_ref** (*ArrayRef* *\|* *None*)

- **copy_in_slot** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*jax.Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*)

- **wait_in_slot** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*jax.Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*)

- **copy_out_slot** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*jax.Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*)

- **wait_out_slot** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*jax.Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*)

- **next_fetch** (*Sequence\[*[*jax.Array*](jax.Array.html#jax.Array "jax.Array")*\]* *\|* *None*)

- **sem_recvs** (*SemaphoreTuple* *\|* *None*)

- **sem_sends** (*SemaphoreTuple* *\|* *None*)

- **tiling** (*Tiling* *\|* *None*)

- **is_trivial_windowing** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **has_allocated_buffer** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

spec[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/pipeline.py#L492-L495)[\#](#jax.experimental.pallas.tpu.BufferedRef.spec "Link to this definition")  
pallas blockspec.

buffer_type[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/pipeline.py#L496-L499)[\#](#jax.experimental.pallas.tpu.BufferedRef.buffer_type "Link to this definition")  
enum indicating whether this is an input, output, or in/out buffered reference.

window_ref[\#](#jax.experimental.pallas.tpu.BufferedRef.window_ref "Link to this definition")  
a multiple-buffer to hold the working and dirty buffers used to copy into and out of. In the case of a BufferedRef targeting a VMEM reference, this simply points to the existing ref.

Type:  
ArrayRef \| None

copy_in_slot[\#](#jax.experimental.pallas.tpu.BufferedRef.copy_in_slot "Link to this definition")  
current slot to copy in for the working buffer.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| [jax.Array](jax.Array.html#jax.Array "jax.Array") \| None

copy_out_slot[\#](#jax.experimental.pallas.tpu.BufferedRef.copy_out_slot "Link to this definition")  
current slot to copy out for the working buffer.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| [jax.Array](jax.Array.html#jax.Array "jax.Array") \| None

wait_in_slot[\#](#jax.experimental.pallas.tpu.BufferedRef.wait_in_slot "Link to this definition")  
current slot to wait in for the working buffer.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| [jax.Array](jax.Array.html#jax.Array "jax.Array") \| None

wait_out_slot[\#](#jax.experimental.pallas.tpu.BufferedRef.wait_out_slot "Link to this definition")  
current slot to wait out for the working buffer.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| [jax.Array](jax.Array.html#jax.Array "jax.Array") \| None

next_fetch[\#](#jax.experimental.pallas.tpu.BufferedRef.next_fetch "Link to this definition")  
Holds the next grid indices to fetch for lookahead. This is the register state used to track the indices within the pipeline loop.

Type:  
Sequence\[[jax.Array](jax.Array.html#jax.Array "jax.Array")\] \| None

sem_recvs[\#](#jax.experimental.pallas.tpu.BufferedRef.sem_recvs "Link to this definition")  
Multiple buffered semaphores for input DMAs.

Type:  
SemaphoreTuple \| None

sem_sends[\#](#jax.experimental.pallas.tpu.BufferedRef.sem_sends "Link to this definition")  
Multiple buffered semaphores for output DMAs.

Type:  
SemaphoreTuple \| None

tiling[\#](#jax.experimental.pallas.tpu.BufferedRef.tiling "Link to this definition")  
The tiling to assume for the buffers.

Type:  
Tiling \| None

is_trivial_windowing[\#](#jax.experimental.pallas.tpu.BufferedRef.is_trivial_windowing "Link to this definition")  
Whether the reference uses trivial windowing.

Type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

has_allocated_buffer[\#](#jax.experimental.pallas.tpu.BufferedRef.has_allocated_buffer "Link to this definition")  
Whether the reference has an allocated buffer due to being in a different memory space than the source ref.

Type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

\_\_init\_\_(*\_spec*, *\_buffer_type*, *\_buffer_count*, *\_grid_rank*, *window_ref*, *copy_in_slot*, *wait_in_slot*, *copy_out_slot*, *wait_out_slot*, *next_fetch*, *sem_recvs*, *sem_sends*, *tiling*, *is_trivial_windowing=False*, *has_allocated_buffer=False*)[\#](#jax.experimental.pallas.tpu.BufferedRef.__init__ "Link to this definition")  
Parameters:  
- **\_spec** (*pallas_core.BlockSpec*)

- **\_buffer_type** (*BufferType*)

- **\_buffer_count** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **\_grid_rank** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

- **window_ref** (*ArrayRef* *\|* *None*)

- **copy_in_slot** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*jax.Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*)

- **wait_in_slot** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*jax.Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*)

- **copy_out_slot** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*jax.Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*)

- **wait_out_slot** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*jax.Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*)

- **next_fetch** (*Sequence\[*[*jax.Array*](jax.Array.html#jax.Array "jax.Array")*\]* *\|* *None*)

- **sem_recvs** (*SemaphoreTuple* *\|* *None*)

- **sem_sends** (*SemaphoreTuple* *\|* *None*)

- **tiling** (*Tiling* *\|* *None*)

- **is_trivial_windowing** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **has_allocated_buffer** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Return type:  
None

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.pallas.tpu.BufferedRef.__init__ "jax.experimental.pallas.tpu.BufferedRef.__init__")(\_spec, \_buffer_type, \_buffer_count, ...) |  |
| `advance_copy_in_slot`(\[predicate\]) | Switch to the next copy slot. |
| `advance_copy_out_slot`(\[predicate\]) | Switch to the next copy slot. |
| `advance_wait_in_slot`(\[predicate\]) | Switch to the next wait slot. |
| `advance_wait_out_slot`(\[predicate\]) | Switch to the next wait slot. |
| `bind_existing_ref`(window_ref, indices) | For handling VMEM references, the pipeline aliases the existing ref. |
| `compute_slice`(grid_indices) | Compute DMA slice from grid indices. |
| `copy_in`(src_ref, grid_indices) | Starts copy of HBM dma slice into the current slot. |
| `copy_out`(dst_ref, grid_indices) | Starts copy of HBM dma slice from the current slot. |
| `create`(spec, dtype_or_type, buffer_type, ...) | Create a BufferedRef. |
| `get_dma_slice`(src_ty, grid_indices) |  |
| `initialize_slots`() | Initializes slots to 0. |
| `input`(spec, dtype_or_type\[, buffer_count\]) |  |
| `input_output`(spec, dtype_or_type\[, buffer_count\]) |  |
| `output`(spec, dtype_or_type\[, buffer_count\]) |  |
| `unbind_refs`() |  |
| `wait_in`(src_ref, grid_indices) | Waits for input copy to finish. |
| `wait_out`(dst_ref, grid_indices) | Waits for output copy to finish. |
| `with_next_fetch`(\[next_fetch\]) |  |
| `with_slot_index`(\[copy_in_slot, ...\]) | Returns a new BufferedRef with the given slot index. |
| `with_spec`(spec) | Returns a new BufferedRef with the given block spec. |

Attributes

|  |  |
|----|----|
| `block_shape` |  |
| `buffer_count` | Returns the number of buffers used for multiple buffering. |
| [`buffer_type`](#jax.experimental.pallas.tpu.BufferedRef.buffer_type "jax.experimental.pallas.tpu.BufferedRef.buffer_type") |  |
| `compute_index` |  |
| `cumulative_copy_in` | The cumulative number of copy_ins issued on this buffer. |
| `cumulative_copy_out` | The cumulative number of copy_outs issued on this buffer. |
| `cumulative_wait_in` | The cumulative number of wait_ins issued on this buffer. |
| `cumulative_wait_out` | The cumulative number of wait_outs issued on this buffer. |
| `current_copy_in_slot` | Index in multiple buffer corresponding to the current slot. |
| `current_copy_out_slot` | Index in multiple buffer corresponding to the current copy slot. |
| `current_ref` | Returns the current working slice of the double-buffer. |
| `current_wait_in_slot` | Index in multiple buffer corresponding to the current wait slot. |
| `current_wait_out_slot` | Index in multiple buffer corresponding to the current wait slot. |
| [`has_allocated_buffer`](#jax.experimental.pallas.tpu.BufferedRef.has_allocated_buffer "jax.experimental.pallas.tpu.BufferedRef.has_allocated_buffer") |  |
| `has_indirect` | Whether any block dimension uses indirect indexing. |
| `is_buffered` | Whether this buffer is multiple-buffered. |
| `is_input` |  |
| `is_input_output` |  |
| `is_manual` |  |
| `is_output` |  |
| [`is_trivial_windowing`](#jax.experimental.pallas.tpu.BufferedRef.is_trivial_windowing "jax.experimental.pallas.tpu.BufferedRef.is_trivial_windowing") |  |
| `next_fetch_indices` | Returns the next grid indices to fetch from if using lookahead. |
| [`spec`](#jax.experimental.pallas.tpu.BufferedRef.spec "jax.experimental.pallas.tpu.BufferedRef.spec") |  |
| `use_lookahead` | Whether this buffer allows lookahead for fetching blocks. |
| [`window_ref`](#jax.experimental.pallas.tpu.BufferedRef.window_ref "jax.experimental.pallas.tpu.BufferedRef.window_ref") |  |
| [`copy_in_slot`](#jax.experimental.pallas.tpu.BufferedRef.copy_in_slot "jax.experimental.pallas.tpu.BufferedRef.copy_in_slot") |  |
| [`wait_in_slot`](#jax.experimental.pallas.tpu.BufferedRef.wait_in_slot "jax.experimental.pallas.tpu.BufferedRef.wait_in_slot") |  |
| [`copy_out_slot`](#jax.experimental.pallas.tpu.BufferedRef.copy_out_slot "jax.experimental.pallas.tpu.BufferedRef.copy_out_slot") |  |
| [`wait_out_slot`](#jax.experimental.pallas.tpu.BufferedRef.wait_out_slot "jax.experimental.pallas.tpu.BufferedRef.wait_out_slot") |  |
| [`next_fetch`](#jax.experimental.pallas.tpu.BufferedRef.next_fetch "jax.experimental.pallas.tpu.BufferedRef.next_fetch") |  |
| [`sem_recvs`](#jax.experimental.pallas.tpu.BufferedRef.sem_recvs "jax.experimental.pallas.tpu.BufferedRef.sem_recvs") |  |
| [`sem_sends`](#jax.experimental.pallas.tpu.BufferedRef.sem_sends "jax.experimental.pallas.tpu.BufferedRef.sem_sends") |  |
| [`tiling`](#jax.experimental.pallas.tpu.BufferedRef.tiling "jax.experimental.pallas.tpu.BufferedRef.tiling") |  |

[](jax.experimental.pallas.tpu.sync_copy.html "previous page")

previous

jax.experimental.pallas.tpu.sync_copy

[](jax.experimental.pallas.tpu.BufferedRefBase.html "next page")

next

jax.experimental.pallas.tpu.BufferedRefBase

Contents

- [`BufferedRef`](#jax.experimental.pallas.tpu.BufferedRef)
  - [`BufferedRef.spec`](#jax.experimental.pallas.tpu.BufferedRef.spec)
  - [`BufferedRef.buffer_type`](#jax.experimental.pallas.tpu.BufferedRef.buffer_type)
  - [`BufferedRef.window_ref`](#jax.experimental.pallas.tpu.BufferedRef.window_ref)
  - [`BufferedRef.copy_in_slot`](#jax.experimental.pallas.tpu.BufferedRef.copy_in_slot)
  - [`BufferedRef.copy_out_slot`](#jax.experimental.pallas.tpu.BufferedRef.copy_out_slot)
  - [`BufferedRef.wait_in_slot`](#jax.experimental.pallas.tpu.BufferedRef.wait_in_slot)
  - [`BufferedRef.wait_out_slot`](#jax.experimental.pallas.tpu.BufferedRef.wait_out_slot)
  - [`BufferedRef.next_fetch`](#jax.experimental.pallas.tpu.BufferedRef.next_fetch)
  - [`BufferedRef.sem_recvs`](#jax.experimental.pallas.tpu.BufferedRef.sem_recvs)
  - [`BufferedRef.sem_sends`](#jax.experimental.pallas.tpu.BufferedRef.sem_sends)
  - [`BufferedRef.tiling`](#jax.experimental.pallas.tpu.BufferedRef.tiling)
  - [`BufferedRef.is_trivial_windowing`](#jax.experimental.pallas.tpu.BufferedRef.is_trivial_windowing)
  - [`BufferedRef.has_allocated_buffer`](#jax.experimental.pallas.tpu.BufferedRef.has_allocated_buffer)
  - [`BufferedRef.__init__()`](#jax.experimental.pallas.tpu.BufferedRef.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
