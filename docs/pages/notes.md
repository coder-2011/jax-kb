- [](index.html)
- Notes

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/notes.rst "Download source file")
-  .pdf

# Notes

# Notes[\#](#notes "Link to this heading")

This section contains shorter notes on topics relevant to using JAX; see also the longer design discussions in [JAX Enhancement Proposals (JEPs)](jep/index.html).

Dependencies and version compatibility:  
- [API compatibility](api_compatibility.html) outlines JAX’s policies with regard to API compatibility across releases.

- [Python and NumPy version support policy](deprecation.html) outlines JAX’s policies with regard to compatibility with Python and NumPy.

Memory and computation usage:  
- [Asynchronous dispatch](async_dispatch.html) describes JAX’s asynchronous dispatch model.

- [Concurrency](concurrency.html) describes how JAX interacts with other Python concurrency.

- [GPU memory allocation](gpu_memory_allocation.html) describes how JAX interacts with memory allocation on GPU.

Programmer guardrails:  
- [Rank promotion warning](rank_promotion_warning.html) describes how to configure [`jax.numpy`](jax.numpy.html#module-jax.numpy "jax.numpy") to avoid implicit rank promotion.

Arrays and data types:  
- [Type promotion semantics](type_promotion.html) describes JAX’s implicit type promotion for functions of two or more values.

- [Default dtypes and the X64 flag](default_dtypes.html) describes how JAX determines the default dtype for array creation functions.

[](building_on_jax.html "previous page")

previous

Building on JAX

[](api_compatibility.html "next page")

next

API compatibility

By The JAX authors

© Copyright 2024, The JAX Authors.\
