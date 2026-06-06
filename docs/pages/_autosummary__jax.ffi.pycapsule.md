- [](../index.html)
- [API Reference](../jax.html)
- [`jax.ffi` module](../jax.ffi.html)
- jax.ffi.pycapsule

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.ffi.pycapsule.rst "Download source file")
-  .pdf

# jax.ffi.pycapsule

## Contents

- [`pycapsule()`](#jax.ffi.pycapsule)

# jax.ffi.pycapsule[\#](#jax-ffi-pycapsule "Link to this heading")

jax.ffi.pycapsule(*funcptr*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/ffi.py#L129-L161)[\#](#jax.ffi.pycapsule "Link to this definition")  
Wrap a ctypes function pointer in a PyCapsule.

The primary use of this function, and the reason why it lives with in the `jax.ffi` submodule, is to wrap function calls from external compiled libraries to be registered as XLA custom calls.

Example usage:

    import ctypes
    import jax
    from jax.lib import xla_client

    libfoo = ctypes.cdll.LoadLibrary('./foo.so')
    xla_client.register_custom_call_target(
        name="bar",
        fn=jax.ffi.pycapsule(libfoo.bar),
        platform=PLATFORM,
        api_version=API_VERSION
    )

Parameters:  
**funcptr** – A function pointer loaded from a dynamic library using `ctypes`.

Returns:  
An opaque `PyCapsule` object wrapping `funcptr`.

[](jax.ffi.ffi_lowering.html "previous page")

previous

jax.ffi.ffi_lowering

[](jax.ffi.register_ffi_target.html "next page")

next

jax.ffi.register_ffi_target

Contents

- [`pycapsule()`](#jax.ffi.pycapsule)

By The JAX authors

© Copyright 2024, The JAX Authors.\
