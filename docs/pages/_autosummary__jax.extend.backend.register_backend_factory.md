- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.backend` module](../jax.extend.backend.html)
- jax.extend.backend.register_backend_factory

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.backend.register_backend_factory.rst "Download source file")
-  .pdf

# jax.extend.backend.register_backend_factory

## Contents

- [`register_backend_factory()`](#jax.extend.backend.register_backend_factory)

# jax.extend.backend.register_backend_factory[\#](#jax-extend-backend-register-backend-factory "Link to this heading")

jax.extend.backend.register_backend_factory(*name*, *factory*, *\**, *priority=0*, *fail_quietly=True*, *experimental=False*, *make_topology=None*, *c_api=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/xla_bridge.py#L312-L325)[\#](#jax.extend.backend.register_backend_factory "Link to this definition")  
Parameters:  
- **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

- **factory** (*BackendFactory*)

- **priority** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **fail_quietly** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **experimental** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **make_topology** (*TopologyFactory* *\|* *None*)

- **c_api** (*Any* *\|* *None*)

Return type:  
None

[](jax.extend.backend.register_backend_cache.html "previous page")

previous

jax.extend.backend.register_backend_cache

[](../jax.extend.core.html "next page")

next

`jax.extend.core` module

Contents

- [`register_backend_factory()`](#jax.extend.backend.register_backend_factory)

By The JAX authors

© Copyright 2024, The JAX Authors.\
