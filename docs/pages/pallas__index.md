- [](../index.html)
- Pallas: a JAX kernel language

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/pallas/index.rst "Download source file")
-  .pdf

# Pallas: a JAX kernel language

# Pallas: a JAX kernel language[\#](#pallas-a-jax-kernel-language "Link to this heading")

Pallas is an extension to JAX that enables writing custom kernels for GPU and TPU. It aims to provide fine-grained control over the generated code, combined with the high-level ergonomics of JAX tracing and the jax.numpy API.

This section contains tutorials, guides and examples for using Pallas. See also the [`jax.experimental.pallas`](../jax.experimental.pallas.html#module-jax.experimental.pallas "jax.experimental.pallas") module API documentation.

Warning

Pallas is experimental and is changing frequently. See the [Pallas Changelog](CHANGELOG.html#pallas-changelog) for the recent changes.

You can expect to encounter errors and unimplemented cases, e.g., when lowering of high-level JAX concepts that would require emulation, or simply because Pallas is still under development.

Guides

- [Pallas Quickstart](quickstart.html)
  - [Hello world in Pallas](quickstart.html#hello-world-in-pallas)
  - [Pallas programming model](quickstart.html#pallas-programming-model)
- [Software Pipelining](pipelining.html)
  - [Memory Hierarchies](pipelining.html#memory-hierarchies)
  - [Pipelining Basics](pipelining.html#pipelining-basics)
  - [Pallas Pipelining API](pipelining.html#pallas-pipelining-api)
  - [Sharp edges](pipelining.html#sharp-edges)
  - [Analyzing the performance](pipelining.html#analyzing-the-performance)
- [Grids and BlockSpecs](grid_blockspec.html)
  - [`grid`, a.k.a. kernels in a loop](grid_blockspec.html#grid-a-k-a-kernels-in-a-loop)
  - [`BlockSpec`, a.k.a. how to chunk up inputs](grid_blockspec.html#blockspec-a-k-a-how-to-chunk-up-inputs)

TPU backend guide

- [Pallas TPU](tpu/index.html)
  - [Writing TPU kernels with Pallas](tpu/details.html)
  - [TPU Pipelining](tpu/pipelining.html)
  - [Matrix Multiplication](tpu/matmul.html)
  - [Scalar Prefetch and Block-Sparse Computation](tpu/sparse.html)
  - [Distributed Computing in Pallas for TPUs](tpu/distributed.html)
  - [Pallas Core-specific Programming](tpu/core_map.html)
  - [SparseCore Kernel Writing](tpu/sparsecore.html)
  - [Pseudo-Random Number Generation](tpu/prng.html)
  - [TPU Hardware Reference](tpu/hardware.html)

Mosaic GPU backend guide

- [Pallas:Mosaic GPU](gpu/index.html)
  - [Writing Mosaic GPU kernels with Pallas](gpu/reference.html)
  - [Mosaic GPU Pipelining](gpu/pipelining.html)
  - [Writing high-performance matrix multiplication kernels for Blackwell](gpu/blackwell_matmul.html)
  - [Collective matrix multiplication](gpu/collective_matmul.html)

Instruction Reference

- [Instruction Reference](../jax.experimental.pallas.html)
  - [Backends](../jax.experimental.pallas.html#backends)
  - [Classes](../jax.experimental.pallas.html#classes)
  - [Functions](../jax.experimental.pallas.html#functions)
  - [Synchronization](../jax.experimental.pallas.html#synchronization)

Design Notes

- [Pallas Design Notes](design/index.html)
  - [Pallas Design](design/design.html)
  - [Pallas Async Operations](design/async_note.html)

Other

- [Pallas Changelog](CHANGELOG.html)

[](../default_dtypes.html "previous page")

previous

Default dtypes and the X64 flag

[](quickstart.html "next page")

next

Pallas Quickstart

By The JAX authors

© Copyright 2024, The JAX Authors.\
