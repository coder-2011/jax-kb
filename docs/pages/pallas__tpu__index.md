- [](../../index.html)
- [Pallas: a JAX kernel language](../index.html)
- Pallas TPU

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../../_sources/pallas/tpu/index.rst "Download source file")
-  .pdf

# Pallas TPU

# Pallas TPU[\#](#pallas-tpu "Link to this heading")

TPU specific documentation.

Guides

- [Writing TPU kernels with Pallas](details.html)
  - [What is a TPU?](details.html#what-is-a-tpu)
  - [Noteworthy properties and restrictions](details.html#noteworthy-properties-and-restrictions)
  - [Supported operations](details.html#supported-operations)
- [TPU Pipelining](pipelining.html)
  - [TPU and its memory spaces](pipelining.html#tpu-and-its-memory-spaces)
  - [TPU-specific Pipelining Features](pipelining.html#tpu-specific-pipelining-features)
- [Matrix Multiplication](matmul.html)
  - [Background](matmul.html#background)
  - [Your first matrix multiplication kernel](matmul.html#your-first-matrix-multiplication-kernel)
  - [Matrix multiplication performance](matmul.html#matrix-multiplication-performance)
  - [Performance of pipelined kernels](matmul.html#performance-of-pipelined-kernels)
  - [Templating the matrix multiplication](matmul.html#templating-the-matrix-multiplication)
  - [Conclusion](matmul.html#conclusion)
- [Scalar Prefetch and Block-Sparse Computation](sparse.html)
  - [Dynamic Block Indexing with Scalar Prefetch](sparse.html#dynamic-block-indexing-with-scalar-prefetch)
  - [Example: Block Dynamic Slice with Scalar Prefetch](sparse.html#example-block-dynamic-slice-with-scalar-prefetch)
  - [Sparse Kernels: Representing Sparse Data](sparse.html#sparse-kernels-representing-sparse-data)
  - [Example: Sparse @ Dense Matrix Multiplication](sparse.html#example-sparse-dense-matrix-multiplication)
  - [Sparse Access Patterns on Dense Data](sparse.html#sparse-access-patterns-on-dense-data)
  - [Example: Dense @ Dense Matrix Multiplication with a Block-Sparse Output Mask](sparse.html#example-dense-dense-matrix-multiplication-with-a-block-sparse-output-mask)
- [Distributed Computing in Pallas for TPUs](distributed.html)
  - [TPU Topologies](distributed.html#tpu-topologies)
  - [Remote Direct Memory Access (RDMA) Model](distributed.html#remote-direct-memory-access-rdma-model)
  - [Advanced Techniques](distributed.html#advanced-techniques)
  - [Final Notes](distributed.html#final-notes)
- [Pallas Core-specific Programming](core_map.html)
  - [Environment setup](core_map.html#environment-setup)
  - [A simple per-core kernel](core_map.html#a-simple-per-core-kernel)
  - [Pipelining with `core_map`](core_map.html#pipelining-with-core-map)
  - [Scalar prefetch](core_map.html#scalar-prefetch)
  - [Mapping over SparseCores](core_map.html#mapping-over-sparsecores)
- [SparseCore Kernel Writing](sparsecore.html)
  - [Hardware overview](sparsecore.html#hardware-overview)
  - [Operations and workloads](sparsecore.html#operations-and-workloads)
  - [Express SparseCore hardware](sparsecore.html#express-sparsecore-hardware)
  - [A basic SparseCore kernel](sparsecore.html#a-basic-sparsecore-kernel)
  - [Pipelining in SparseCore kernels](sparsecore.html#pipelining-in-sparsecore-kernels)
  - [Overlapping TensorCore and SparseCore](sparsecore.html#overlapping-tensorcore-and-sparsecore)
  - [Gather and scatter](sparsecore.html#gather-and-scatter)
  - [Benchmark against TensorCore](sparsecore.html#benchmark-against-tensorcore)
- [Pseudo-Random Number Generation](prng.html)
  - [Using the `jax.random` API](prng.html#using-the-jax-random-api)
  - [Using the hardware PRNG](prng.html#using-the-hardware-prng)
  - [Block-invariant sampling](prng.html#block-invariant-sampling)
- [TPU Hardware Reference](hardware.html)

[](../grid_blockspec.html "previous page")

previous

Grids and BlockSpecs

[](details.html "next page")

next

Writing TPU kernels with Pallas

By The JAX authors

© Copyright 2024, The JAX Authors.\
