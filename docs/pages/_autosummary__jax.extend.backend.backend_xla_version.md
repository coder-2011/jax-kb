- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.backend` module](../jax.extend.backend.html)
- jax.extend.backend.backend_xla_version

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.backend.backend_xla_version.rst "Download source file")
-  .pdf

# jax.extend.backend.backend_xla_version

## Contents

- [`backend_xla_version()`](#jax.extend.backend.backend_xla_version)

# jax.extend.backend.backend_xla_version[\#](#jax-extend-backend-backend-xla-version "Link to this heading")

jax.extend.backend.backend_xla_version(*platform=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/xla_bridge.py#L1060-L1070)[\#](#jax.extend.backend.backend_xla_version "Link to this definition")  
Returns the XLA version of the backend.

Returns None if the backend does not use PJRT C API or does not have xla_version in the plugin attributes. This method can be used to skip features that are not available before certain xla_version if the backend is a plugin and uses xla_version.

Return type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| None

[](jax.extend.backend.backends.html "previous page")

previous

jax.extend.backend.backends

[](jax.extend.backend.clear_backends.html "next page")

next

jax.extend.backend.clear_backends

Contents

- [`backend_xla_version()`](#jax.extend.backend.backend_xla_version)

By The JAX authors

© Copyright 2024, The JAX Authors.\
