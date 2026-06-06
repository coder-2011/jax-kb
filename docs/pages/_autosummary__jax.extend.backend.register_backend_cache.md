- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.backend` module](../jax.extend.backend.html)
- jax.extend.backend.register_backend_cache

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.backend.register_backend_cache.rst "Download source file")
-  .pdf

# jax.extend.backend.register_backend_cache

## Contents

- [`register_backend_cache()`](#jax.extend.backend.register_backend_cache)

# jax.extend.backend.register_backend_cache[\#](#jax-extend-backend-register-backend-cache "Link to this heading")

jax.extend.backend.register_backend_cache(*cache*, *for_what*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/util.py#L292-L302)[\#](#jax.extend.backend.register_backend_cache "Link to this definition")  
Registers a cache with JAX’s cache management.

Parameters:  
- **cache** (*Any*) – an object supporting cache_clear(), cache_info(), and cache_keys(), like the result of functools.lru_cache().

- **for_what** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – a string to identify what this cache is used for. This is used for debugging.

[](jax.extend.backend.ifrt_proxy.html "previous page")

previous

jax.extend.backend.ifrt_proxy

[](jax.extend.backend.register_backend_factory.html "next page")

next

jax.extend.backend.register_backend_factory

Contents

- [`register_backend_cache()`](#jax.extend.backend.register_backend_cache)

By The JAX authors

© Copyright 2024, The JAX Authors.\
