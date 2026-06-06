- [](index.html)
- Resources and Advanced Guides

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/advanced_guides.rst "Download source file")
-  .pdf

# Resources and Advanced Guides

# Resources and Advanced Guides[\#](#resources-and-advanced-guides "Link to this heading")

This section contains examples and tutorials on more advanced topics, such as multi-core computation, automatic differentiation, and custom operations.

Parallel computation

- [Manual parallelism with `shard_map`](notebooks/shard_map.html)
- [Device-local array layout control](notebooks/layout.html)
- [JAX Memories and Host Offloading](notebooks/host-offloading.html)
- [Optimizer State Offloading](notebooks/host-offloading.html#optimizer-state-offloading)
- [Introduction to multi-controller JAX (aka multi-process/multi-host JAX)](multi_process.html)
- [Fault Tolerant Distributed JAX](fault_tolerance.html)
- [Distributed data loading](distributed_data_loading.html)
- [Colocated Python](notebooks/colocated-python.html)

Machine learning

- [The Training Cookbook](the-training-cookbook.html)

Automatic differentiation

- [The Autodiff Cookbook](notebooks/autodiff_cookbook.html)
- [Control autodiff’s saved values with `jax.checkpoint` (aka `jax.remat`)](notebooks/autodiff_remat.html)
- [Advanced Automatic Differentiation](advanced_autodiff.html)

Errors and debugging

- [Errors](errors.html)
- [Introduction to debugging](debugging.html)
- [Debugging runtime values](debugging/index.html)
- [Transfer guard](transfer_guard.html)

Pytrees

- [Custom pytree nodes](custom_pytrees.html)

Performance optimizations

- [Persistent compilation cache](persistent_compilation_cache.html)
- [Buffer donation](buffer_donation.html)
- [GPU performance tips](gpu_performance_tips.html)

Performance benchmarking and profiling

- [Benchmarking JAX code](benchmarking.html)
- [Profiling computation](profiling.html)
- [Profiling device memory](device_memory_profiling.html)

Non-functional programming

- [`Ref`: mutable arrays for data plumbing and memory control](array_refs.html)

External Callbacks

- [External callbacks](external-callbacks.html)

FFI

- [Foreign function interface (FFI)](ffi.html)
- [Writing High-Performance GPU Kernels with CuTe DSL and JAX](notebooks/cute_dsl_jax.html)

Modeling workflows

- [Gradient checkpointing with `jax.checkpoint` (`jax.remat`)](gradient-checkpointing.html)
- [Ahead-of-time lowering and compilation](aot.html)
- [Exporting and serialization](export/index.html)

Example applications

- [Training a simple neural network, with tensorflow/datasets data loading](notebooks/neural_network_with_tfds_data.html)
- [Training a simple neural network, with PyTorch data loading](notebooks/Neural_Network_and_Data_Loading.html)
- [Autobatching for Bayesian inference](notebooks/vmapped_log_probs.html)

Deep dives

- [Generalized convolutions in JAX](notebooks/convolutions.html)
- [XLA compiler flags](xla_flags.html)
- [JAX Internals: primitives](jax-primitives.html)
- [JAX internals: The jaxpr language](jaxpr.html)

[](key-concepts.html "previous page")

previous

Key concepts

[](notebooks/shard_map.html "next page")

next

Manual parallelism with `shard_map`

By The JAX authors

© Copyright 2024, The JAX Authors.\
