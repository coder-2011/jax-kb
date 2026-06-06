- [](../index.html)
- [API Reference](../jax.html)
- [`jax.ffi` module](../jax.ffi.html)
- jax.ffi.register_ffi_type

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.ffi.register_ffi_type.rst "Download source file")
-  .pdf

# jax.ffi.register_ffi_type

## Contents

- [`register_ffi_type()`](#jax.ffi.register_ffi_type)

# jax.ffi.register_ffi_type[\#](#jax-ffi-register-ffi-type "Link to this heading")

jax.ffi.register_ffi_type(*name*, *type_registration*, *platform='cpu'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/ffi.py#L100-L115)[\#](#jax.ffi.register_ffi_type "Link to this definition")  
Registers a custom type for a FFI target.

Parameters:  
- **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – the name of the type. This name must be unique within the process.

- **type_registration** (*TypeRegistration*) – a `TypeRegistration` defining the external type.

- **platform** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – the target platform.

Return type:  
None

[](jax.ffi.register_ffi_target.html "previous page")

previous

jax.ffi.register_ffi_target

[](../jax.flatten_util.html "next page")

next

`jax.flatten_util` module

Contents

- [`register_ffi_type()`](#jax.ffi.register_ffi_type)

By The JAX authors

© Copyright 2024, The JAX Authors.\
