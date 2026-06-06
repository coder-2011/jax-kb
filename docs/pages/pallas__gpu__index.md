- [](../../index.html)
- [Pallas: a JAX kernel language](../index.html)
- Pallas:Mosaic GPU

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../../_sources/pallas/gpu/index.rst "Download source file")
-  .pdf

# Pallas:Mosaic GPU

# Pallas:Mosaic GPU[\#](#pallas-mosaic-gpu "Link to this heading")

Backend specific documentation for the Mosaic GPU backend.

Reference documentation

- [Writing Mosaic GPU kernels with Pallas](reference.html)
  - [What is a GPU?](reference.html#what-is-a-gpu)
  - [Array layouts and memory reference transforms](reference.html#array-layouts-and-memory-reference-transforms)
  - [MMA (TensorCore)](reference.html#mma-tensorcore)
  - [Using `core_map`](reference.html#using-core-map)
  - [Synchronization structures and primitives](reference.html#synchronization-structures-and-primitives)
  - [Cluster launch control](reference.html#cluster-launch-control)
  - [Asynchronous copies](reference.html#asynchronous-copies)
  - [Inline Mosaic GPU](reference.html#inline-mosaic-gpu)
  - [Compiler parameters](reference.html#compiler-parameters)
  - [Debugging](reference.html#debugging)
  - [Calling kernels from PyTorch](reference.html#calling-kernels-from-pytorch)
- [Mosaic GPU Pipelining](pipelining.html)
  - [Pipelining with Mosaic GPU](pipelining.html#pipelining-with-mosaic-gpu)
  - [GPU Memory Spaces](pipelining.html#gpu-memory-spaces)
  - [Example: Matmul Kernel on Hopper GPUs](pipelining.html#example-matmul-kernel-on-hopper-gpus)
  - [Warp Specialization](pipelining.html#warp-specialization)
  - [Example: Matrix Multiplication with Warp Specialization](pipelining.html#example-matrix-multiplication-with-warp-specialization)
- [Writing high-performance matrix multiplication kernels for Blackwell](blackwell_matmul.html)
  - [0. Basic kernel](blackwell_matmul.html#basic-kernel)
  - [1. Warp specialization](blackwell_matmul.html#warp-specialization)
  - [2. Tiled epilogue](blackwell_matmul.html#tiled-epilogue)
  - [3. Collective (2CTA) MMA](blackwell_matmul.html#collective-2cta-mma)
  - [4. Persistent kernel](blackwell_matmul.html#persistent-kernel)
  - [5. Dedicated epilogue warpgroup](blackwell_matmul.html#dedicated-epilogue-warpgroup)
  - [6. Grid tiling](blackwell_matmul.html#grid-tiling)
  - [Final kernel](blackwell_matmul.html#final-kernel)
- [Collective matrix multiplication](collective_matmul.html)
  - [Algorithm overview: Ring All-Gather](collective_matmul.html#algorithm-overview-ring-all-gather)
  - [Pallas primitives for inter-device communication](collective_matmul.html#pallas-primitives-for-inter-device-communication)
  - [Implementation with Pallas](collective_matmul.html#implementation-with-pallas)
  - [Integrating the kernel with JAX](collective_matmul.html#integrating-the-kernel-with-jax)

[](../tpu/hardware.html "previous page")

previous

TPU Hardware Reference

[](reference.html "next page")

next

Writing Mosaic GPU kernels with Pallas

By The JAX authors

© Copyright 2024, The JAX Authors.\
