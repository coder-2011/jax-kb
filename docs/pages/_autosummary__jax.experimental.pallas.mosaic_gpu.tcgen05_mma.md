- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.tcgen05_mma

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.tcgen05_mma.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.tcgen05_mma

## Contents

- [`tcgen05_mma()`](#jax.experimental.pallas.mosaic_gpu.tcgen05_mma)

# jax.experimental.pallas.mosaic_gpu.tcgen05_mma[\#](#jax-experimental-pallas-mosaic-gpu-tcgen05-mma "Link to this heading")

jax.experimental.pallas.mosaic_gpu.tcgen05_mma(*acc*, *a*, *b*, *barrier=None*, *\**, *a_scale=None*, *b_scale=None*, *a_sparse_metadata=None*, *accumulate=True*, *collective_axis=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L1991-L2146)[\#](#jax.experimental.pallas.mosaic_gpu.tcgen05_mma "Link to this definition")  
Asynchronous matrix-multiply accumulate for TensorCore gen 5 (Blackwell).

If run in collective mode, `acc`, `a` (LHS), and `b` (RHS) should correspond to half of the total inputs to the MMA, where `acc` and `a` (LHS) are split in half along the rows and `b` (RHS) is split along the columns like so:

    -----------    -----------   -----------
    |  ACC1   |    |  LHS1   |   |    |    |
    ----------- += ----------- @ |RHS1|RHS2|
    |  ACC2   |    |  LHS2   |   |    |    |
    -----------    -----------   -----------

To use the block-scaled matrix-multiply, provide `a_scale` and `b_scale` operands (they must be both present or both unspecified).

Parameters:  
- **acc** (*\_Ref*) – The accumulator. Must be a TMEM Ref.

- **a** (*\_Ref*) – The left-hand side. Must be a TMEM/SMEM Ref.

- **b** (*\_Ref*) – The right-hand side. Must be an SMEM Ref.

- **barrier** (*\_Ref* *\|* *None*) – Optional barrier Ref for synchronizing with the tensor core. Must have orders_tensor_core set to True. If not specified, the MMA completion should be explicitly observed by calling [`jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive()`](jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive.html#jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive "jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive")

- **a_scale** (*\_Ref* *\|* *None*) – An optional scale for the `a` operand. Must be a TMEM Ref if present.

- **b_scale** (*\_Ref* *\|* *None*) – An optional scale for the `b` operand. Must be a TMEM Ref if present.

- **a_sparse_metadata** (*\_Ref* *\|* *None*) – An optional sparse metadata for the `a` operand. Must be a TMEM Ref if present.

- **accumulate** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*jax.Array*](jax.Array.html#jax.Array "jax.Array")) – Whether to accumulate into acc or overwrite it.

- **collective_axis** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – The name of the cluster axis along which to perform a collective MMA. The cluster axis should have a size of exactly 2, and must be on the minormost cluster axis.

[](jax.experimental.pallas.mosaic_gpu.wgmma_wait.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.wgmma_wait

[](jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive.html "next page")

next

jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive

Contents

- [`tcgen05_mma()`](#jax.experimental.pallas.mosaic_gpu.tcgen05_mma)

By The JAX authors

© Copyright 2024, The JAX Authors.\
