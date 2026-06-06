- [](index.html)
- [API Reference](jax.html)
- [`jax.extend` module](jax.extend.html)
- `jax.extend.backend` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.extend.backend.rst "Download source file")
-  .pdf

# jax.extend.backend module

# `jax.extend.backend` module[\#](#module-jax.extend.backend "Link to this heading")

|  |  |
|----|----|
| [`backends`](_autosummary/jax.extend.backend.backends.html#jax.extend.backend.backends "jax.extend.backend.backends")() |  |
| [`backend_xla_version`](_autosummary/jax.extend.backend.backend_xla_version.html#jax.extend.backend.backend_xla_version "jax.extend.backend.backend_xla_version")(\[platform\]) | Returns the XLA version of the backend. |
| [`clear_backends`](_autosummary/jax.extend.backend.clear_backends.html#jax.extend.backend.clear_backends "jax.extend.backend.clear_backends")(\[\_crash\]) | Clear all backend clients so that new backend clients can be created later. |
| [`get_backend`](_autosummary/jax.extend.backend.get_backend.html#jax.extend.backend.get_backend "jax.extend.backend.get_backend")(\[platform\]) |  |
| [`get_compile_options`](_autosummary/jax.extend.backend.get_compile_options.html#jax.extend.backend.get_compile_options "jax.extend.backend.get_compile_options")(num_replicas, num_partitions) | Returns the compile options to use, as derived from flag values. |
| [`get_default_device`](_autosummary/jax.extend.backend.get_default_device.html#jax.extend.backend.get_default_device "jax.extend.backend.get_default_device")() |  |
| [`ifrt_proxy`](_autosummary/jax.extend.backend.ifrt_proxy.html#module-jax.extend.backend.ifrt_proxy "jax.extend.backend.ifrt_proxy") |  |
| [`register_backend_cache`](_autosummary/jax.extend.backend.register_backend_cache.html#jax.extend.backend.register_backend_cache "jax.extend.backend.register_backend_cache")(cache, for_what) | Registers a cache with JAX's cache management. |
| [`register_backend_factory`](_autosummary/jax.extend.backend.register_backend_factory.html#jax.extend.backend.register_backend_factory "jax.extend.backend.register_backend_factory")(name, factory, \*\[, ...\]) |  |

[](jax.extend.html "previous page")

previous

`jax.extend` module

[](_autosummary/jax.extend.backend.backends.html "next page")

next

jax.extend.backend.backends

By The JAX authors

© Copyright 2024, The JAX Authors.\
