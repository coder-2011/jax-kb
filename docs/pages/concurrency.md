- [](index.html)
- [Notes](notes.html)
- Concurrency

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/concurrency.rst "Download source file")
-  .pdf

# Concurrency

# Concurrency[\#](#concurrency "Link to this heading")

JAX has limited support for Python concurrency.

Clients may call JAX APIs (e.g., [`jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit") or [`grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad")) concurrently from separate Python threads.

It is not permitted to manipulate JAX trace values concurrently from multiple threads. In other words, while it is permissible to call functions that use JAX tracing (e.g., [`jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit")) from multiple threads, you must not use threading to manipulate JAX values inside the implementation of the function f that is passed to [`jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit"). The most likely outcome if you do this is a mysterious error from JAX.

In multi-controller JAX, different processes must apply the same JAX operations in the same order on a given device. If you are using threads with multi-controller JAX, you can use the `thread_guard()` context manager to detect cases where threads may schedule operations in different orders in different processes, leading to non-deterministic crashes. When the thread guard is set, an error will be raised at runtime if a JAX operation is called from a thread other than the one in which the thread guard was set.

[](async_dispatch.html "previous page")

previous

Asynchronous dispatch

[](gpu_memory_allocation.html "next page")

next

GPU memory allocation

By The JAX authors

© Copyright 2024, The JAX Authors.\
