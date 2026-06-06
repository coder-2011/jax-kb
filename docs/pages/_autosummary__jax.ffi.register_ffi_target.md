- [](../index.html)
- [API Reference](../jax.html)
- [`jax.ffi` module](../jax.ffi.html)
- jax.ffi.register_ffi_target

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.ffi.register_ffi_target.rst "Download source file")
-  .pdf

# jax.ffi.register_ffi_target

## Contents

- [`register_ffi_target()`](#jax.ffi.register_ffi_target)

# jax.ffi.register_ffi_target[\#](#jax-ffi-register-ffi-target "Link to this heading")

jax.ffi.register_ffi_target(*name*, *fn*, *platform='cpu'*, *api_version=1*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/ffi.py#L46-L69)[\#](#jax.ffi.register_ffi_target "Link to this definition")  
Registers a foreign function target.

Parameters:  
- **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – the name of the target.

- **fn** (*Any*) – a `PyCapsule` object containing the function pointer, or a `dict` where the keys are FFI stage names (e.g. “execute”) and the values are `PyCapsule` objects containing a pointer to the handler for that stage.

- **platform** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – the target platform.

- **api_version** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the XLA custom call API version to use. Supported versions are: 1 (default) for the typed FFI or 0 for the earlier “custom call” API.

- **kwargs** (*Any*) – any extra keyword arguments are passed directly to `register_custom_call_target()` for more advanced use cases.

Return type:  
None

[](jax.ffi.pycapsule.html "previous page")

previous

jax.ffi.pycapsule

[](jax.ffi.register_ffi_type.html "next page")

next

jax.ffi.register_ffi_type

Contents

- [`register_ffi_target()`](#jax.ffi.register_ffi_target)

By The JAX authors

© Copyright 2024, The JAX Authors.\
