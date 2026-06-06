- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.backend` module](../jax.extend.backend.html)
- jax.extend.backend.get_compile_options

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.backend.get_compile_options.rst "Download source file")
-  .pdf

# jax.extend.backend.get_compile_options

## Contents

- [`get_compile_options()`](#jax.extend.backend.get_compile_options)

# jax.extend.backend.get_compile_options[\#](#jax-extend-backend-get-compile-options "Link to this heading")

jax.extend.backend.get_compile_options(*num_replicas*, *num_partitions*, *device_assignment=None*, *env_options_overrides=None*, *fdo_profile=None*, *detailed_logging=True*, *backend=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/compiler.py#L139-L290)[\#](#jax.extend.backend.get_compile_options "Link to this definition")  
Returns the compile options to use, as derived from flag values.

Parameters:  
- **num_replicas** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Number of replicas for which to compile.

- **num_partitions** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Number of partitions for which to compile.

- **device_assignment** – Optional ndarray of jax devices indicating the assignment of logical replicas to physical devices (default inherited from xla_client.CompileOptions). Must be consistent with num_replicas and num_partitions.

- **env_options_overrides** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]* *\|* *None*) – dict of additional options parsed by the compiler

- **fdo_profile** ([*bytes*](https://docs.python.org/3/library/stdtypes.html#bytes "(in Python v3.14)") *\|* *None*) – Optional profile for feedback-directed optimization passed to XLA.

- **detailed_logging** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Is this an “interesting” computation about which XLA would be wise to log compilation information?

- **backend** (*xc.Client* *\|* *None*) – the client, if available.

Return type:  
xc.CompileOptions

[](jax.extend.backend.get_backend.html "previous page")

previous

jax.extend.backend.get_backend

[](jax.extend.backend.get_default_device.html "next page")

next

jax.extend.backend.get_default_device

Contents

- [`get_compile_options()`](#jax.extend.backend.get_compile_options)

By The JAX authors

© Copyright 2024, The JAX Authors.\
