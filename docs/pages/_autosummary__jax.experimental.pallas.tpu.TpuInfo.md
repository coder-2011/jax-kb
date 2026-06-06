- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.TpuInfo

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.TpuInfo.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.TpuInfo

## Contents

- [`TpuInfo`](#jax.experimental.pallas.tpu.TpuInfo)
  - [`TpuInfo.__init__()`](#jax.experimental.pallas.tpu.TpuInfo.__init__)

# jax.experimental.pallas.tpu.TpuInfo[\#](#jax-experimental-pallas-tpu-tpuinfo "Link to this heading")

*class* jax.experimental.pallas.tpu.TpuInfo(*\**, *chip_version*, *generation*, *num_cores*, *num_lanes*, *num_sublanes*, *mxu_column_size*, *num_mxus*, *num_accumulators*, *vmem_capacity_bytes*, *cmem_capacity_bytes*, *smem_capacity_bytes*, *hbm_capacity_bytes*, *mem_bw_bytes_per_second*, *bf16_ops_per_second*, *int8_ops_per_second*, *fp8_ops_per_second*, *int4_ops_per_second*, *sparse_core=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/tpu_info.py#L148-L270)[\#](#jax.experimental.pallas.tpu.TpuInfo "Link to this definition")  
TPU hardware information.

Note that all information is per-TensorCore so you would need to multiply by num_cores to obtain the total for the chip.

Parameters:  
- **chip_version** (*ChipVersionBase*)

- **generation** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **num_cores** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **num_lanes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **num_sublanes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **mxu_column_size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **num_mxus** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **num_accumulators** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **vmem_capacity_bytes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **cmem_capacity_bytes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **smem_capacity_bytes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **hbm_capacity_bytes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **mem_bw_bytes_per_second** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **bf16_ops_per_second** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **int8_ops_per_second** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **fp8_ops_per_second** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **int4_ops_per_second** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **sparse_core** (*SparseCoreInfo* *\|* *None*)

\_\_init\_\_(*\**, *chip_version*, *generation*, *num_cores*, *num_lanes*, *num_sublanes*, *mxu_column_size*, *num_mxus*, *num_accumulators*, *vmem_capacity_bytes*, *cmem_capacity_bytes*, *smem_capacity_bytes*, *hbm_capacity_bytes*, *mem_bw_bytes_per_second*, *bf16_ops_per_second*, *int8_ops_per_second*, *fp8_ops_per_second*, *int4_ops_per_second*, *sparse_core=None*)[\#](#jax.experimental.pallas.tpu.TpuInfo.__init__ "Link to this definition")  
Parameters:  
- **chip_version** (*ChipVersionBase*)

- **generation** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **num_cores** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **num_lanes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **num_sublanes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **mxu_column_size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **num_mxus** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **num_accumulators** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **vmem_capacity_bytes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **cmem_capacity_bytes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **smem_capacity_bytes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **hbm_capacity_bytes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **mem_bw_bytes_per_second** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **bf16_ops_per_second** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **int8_ops_per_second** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **fp8_ops_per_second** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **int4_ops_per_second** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **sparse_core** (*SparseCoreInfo* *\|* *None*)

Return type:  
None

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.pallas.tpu.TpuInfo.__init__ "jax.experimental.pallas.tpu.TpuInfo.__init__")(\*, chip_version, generation, ...\[, ...\]) |  |
| `get_sublane_tiling`(dtype) | Returns the sublane tiling for the given itemsize. |
| `is_matmul_supported`(lhs_dtype, rhs_dtype) | Returns whether the chip natively supports matmul on the given input dtypes (no casting needed). |

Attributes

|  |  |
|----|----|
| `is_lite` |  |
| `is_megacore` | Returns True if the chip is configured in Megacore mode. |
| `is_split_chip` | Returns True if the chip is a multi-core chip being used in single-core mode. |
| `sparse_core` |  |
| `chip_version` |  |
| `generation` |  |
| `num_cores` |  |
| `num_lanes` |  |
| `num_sublanes` |  |
| `mxu_column_size` |  |
| `num_mxus` |  |
| `num_accumulators` |  |
| `vmem_capacity_bytes` |  |
| `cmem_capacity_bytes` |  |
| `smem_capacity_bytes` |  |
| `hbm_capacity_bytes` |  |
| `mem_bw_bytes_per_second` |  |
| `bf16_ops_per_second` |  |
| `int8_ops_per_second` |  |
| `fp8_ops_per_second` |  |
| `int4_ops_per_second` |  |

[](jax.experimental.pallas.tpu.SemaphoreType.html "previous page")

previous

jax.experimental.pallas.tpu.SemaphoreType

[](jax.experimental.pallas.tpu.load.html "next page")

next

jax.experimental.pallas.tpu.load

Contents

- [`TpuInfo`](#jax.experimental.pallas.tpu.TpuInfo)
  - [`TpuInfo.__init__()`](#jax.experimental.pallas.tpu.TpuInfo.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
